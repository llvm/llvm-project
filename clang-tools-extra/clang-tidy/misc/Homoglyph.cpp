//===--- MisleadingBidirectional.cpp - clang-tidy--------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Homoglyph.h"

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/Support/ConvertUTF.h"

namespace {
// Preprocessed version of
// https://www.unicode.org/Public/security/latest/confusables.txt
//
// This contains a sorted array of { UTF32 codepoint; UTF32 values[N];}
#include "Confusables.inc"
} // namespace

namespace clang {
namespace tidy {
namespace misc {

Homoglyph::Homoglyph(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {}

Homoglyph::~Homoglyph() = default;

/**
 * Build a skeleton out of the Original identifier, following the algorithm
 * described in http://www.unicode.org/reports/tr39/#def-skeleton
 */
std::string Homoglyph::skeleton(StringRef Name) {
  std::string SName = Name.str();
  std::string Skeleton;
  Skeleton.reserve(1 + Name.size());

  char const *Curr = SName.c_str();
  char const *End = Curr + SName.size();
  while (Curr < End) {

    char const *Prev = Curr;
    llvm::UTF32 CodePoint;
    llvm::ConversionResult Result = llvm::convertUTF8Sequence(
        (const llvm::UTF8 **)&Curr, (const llvm::UTF8 *)End, &CodePoint,
        llvm::strictConversion);
    if (Result != llvm::conversionOK) {
      llvm::errs() << "Unicode conversion issue\n";
      break;
    }

    StringRef Key(Prev, Curr - Prev);
    auto Where = std::lower_bound(
        std::begin(ConfusableEntries), std::end(ConfusableEntries), CodePoint,
        [](decltype(ConfusableEntries[0]) x, llvm::UTF32 y) {
          return x.codepoint < y;
        });
    if (Where == std::end(ConfusableEntries) || CodePoint != Where->codepoint) {
      Skeleton.append(Prev, Curr);
    } else {
      llvm::UTF8 Buffer[32];
      llvm::UTF8 *BufferStart = std::begin(Buffer);
      llvm::UTF8 *IBuffer = BufferStart;
      const llvm::UTF32 *ValuesStart = std::begin(Where->values);
      const llvm::UTF32 *ValuesEnd =
          std::find(std::begin(Where->values), std::end(Where->values), '\0');
      if (llvm::ConvertUTF32toUTF8(&ValuesStart, ValuesEnd, &IBuffer,
                                   std::end(Buffer), llvm::strictConversion) !=
          llvm::conversionOK) {
        llvm::errs() << "Unicode conversion issue\n";
        break;
      }
      Skeleton.append((char *)BufferStart, (char *)IBuffer);
    }
  }
  return Skeleton;
}

void Homoglyph::check(const ast_matchers::MatchFinder::MatchResult &Result) {
  if (const auto *ND = Result.Nodes.getNodeAs<NamedDecl>("nameddecl")) {
    StringRef NDName = ND->getName();
    auto &Mapped = Mapper[skeleton(NDName)];
    auto *NDDecl = ND->getDeclContext();
    for (auto *OND : Mapped) {
      if (!NDDecl->isDeclInLexicalTraversal(OND) &&
          !OND->getDeclContext()->isDeclInLexicalTraversal(ND))
        continue;
      if (OND->getName() != NDName) {
        diag(OND->getLocation(), "%0 is confusable with %1")
            << OND->getName() << NDName;
        diag(ND->getLocation(), "other definition found here",
             DiagnosticIDs::Note);
      }
    }
    Mapped.push_back(ND);
  }
}

void Homoglyph::registerMatchers(ast_matchers::MatchFinder *Finder) {
  Finder->addMatcher(ast_matchers::namedDecl().bind("nameddecl"), this);
}

} // namespace misc
} // namespace tidy
} // namespace clang

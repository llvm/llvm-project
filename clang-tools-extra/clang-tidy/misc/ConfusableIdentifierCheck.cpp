//===--- ConfusableIdentifierCheck.cpp -
// clang-tidy--------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ConfusableIdentifierCheck.h"

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

ConfusableIdentifierCheck::ConfusableIdentifierCheck(StringRef Name,
                                                     ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {}

ConfusableIdentifierCheck::~ConfusableIdentifierCheck() = default;

// Build a skeleton out of the Original identifier, inspired by the algorithm
// described in http://www.unicode.org/reports/tr39/#def-skeleton
//
// FIXME: TR39 mandates:
//
// For an input string X, define skeleton(X) to be the following transformation
// on the string:
//
// 1. Convert X to NFD format, as described in [UAX15].
// 2. Concatenate the prototypes for each character in X according to the
// specified data, producing a string of exemplar characters.
// 3. Reapply NFD.
//
// We're skipping 1. and 3. for the sake of simplicity, but this can lead to
// false positive.

std::string ConfusableIdentifierCheck::skeleton(StringRef Name) {
  using namespace llvm;
  std::string SName = Name.str();
  std::string Skeleton;
  Skeleton.reserve(1 + Name.size());

  const char *Curr = SName.c_str();
  const char *End = Curr + SName.size();
  while (Curr < End) {

    const char *Prev = Curr;
    UTF32 CodePoint;
    ConversionResult Result = convertUTF8Sequence(
        reinterpret_cast<const UTF8 **>(&Curr),
        reinterpret_cast<const UTF8 *>(End), &CodePoint, strictConversion);
    if (Result != conversionOK) {
      errs() << "Unicode conversion issue\n";
      break;
    }

    StringRef Key(Prev, Curr - Prev);
    auto Where = llvm::lower_bound(ConfusableEntries, CodePoint,
                                   [](decltype(ConfusableEntries[0]) x,
                                      UTF32 y) { return x.codepoint < y; });
    if (Where == std::end(ConfusableEntries) || CodePoint != Where->codepoint) {
      Skeleton.append(Prev, Curr);
    } else {
      UTF8 Buffer[32];
      UTF8 *BufferStart = std::begin(Buffer);
      UTF8 *IBuffer = BufferStart;
      const UTF32 *ValuesStart = std::begin(Where->values);
      const UTF32 *ValuesEnd = llvm::find(Where->values, '\0');
      if (ConvertUTF32toUTF8(&ValuesStart, ValuesEnd, &IBuffer,
                             std::end(Buffer),
                             strictConversion) != conversionOK) {
        errs() << "Unicode conversion issue\n";
        break;
      }
      Skeleton.append((char *)BufferStart, (char *)IBuffer);
    }
  }
  return Skeleton;
}

static bool mayShadowImpl(const NamedDecl *ND0, const NamedDecl *ND1) {
  const DeclContext *DC0 = ND0->getDeclContext()->getPrimaryContext();
  const DeclContext *DC1 = ND1->getDeclContext()->getPrimaryContext();

  if (isa<TemplateTypeParmDecl>(ND0) || isa<TemplateTypeParmDecl>(ND0))
    return true;

  while (DC0->isTransparentContext())
    DC0 = DC0->getParent();
  while (DC1->isTransparentContext())
    DC1 = DC1->getParent();

  if (DC0->Equals(DC1))
    return true;

  return false;
}

static bool isMemberOf(const NamedDecl *ND, const CXXRecordDecl *RD) {
  const DeclContext *NDParent = ND->getDeclContext();
  if (!NDParent || !isa<CXXRecordDecl>(NDParent))
    return false;
  if (NDParent == RD)
    return true;
  return !RD->forallBases(
      [NDParent](const CXXRecordDecl *Base) { return NDParent != Base; });
}

static bool mayShadow(const NamedDecl *ND0, const NamedDecl *ND1) {

  const DeclContext *DC0 = ND0->getDeclContext()->getPrimaryContext();
  const DeclContext *DC1 = ND1->getDeclContext()->getPrimaryContext();

  if (const CXXRecordDecl *RD0 = dyn_cast<CXXRecordDecl>(DC0)) {
    RD0 = RD0->getDefinition();
    if (RD0 && ND1->getAccess() != AS_private && isMemberOf(ND1, RD0))
      return true;
  }
  if (const CXXRecordDecl *RD1 = dyn_cast<CXXRecordDecl>(DC1)) {
    RD1 = RD1->getDefinition();
    if (RD1 && ND0->getAccess() != AS_private && isMemberOf(ND0, RD1))
      return true;
  }

  if (DC0->Encloses(DC1))
    return mayShadowImpl(ND0, ND1);
  if (DC1->Encloses(DC0))
    return mayShadowImpl(ND1, ND0);
  return false;
}

void ConfusableIdentifierCheck::check(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (const auto *ND = Result.Nodes.getNodeAs<NamedDecl>("nameddecl")) {
    if (IdentifierInfo *NDII = ND->getIdentifier()) {
      StringRef NDName = NDII->getName();
      llvm::SmallVector<const NamedDecl *> &Mapped = Mapper[skeleton(NDName)];
      for (const NamedDecl *OND : Mapped) {
        const IdentifierInfo *ONDII = OND->getIdentifier();
        if (mayShadow(ND, OND)) {
          StringRef ONDName = ONDII->getName();
          if (ONDName != NDName) {
            diag(ND->getLocation(), "%0 is confusable with %1") << ND << OND;
            diag(OND->getLocation(), "other declaration found here",
                 DiagnosticIDs::Note);
          }
        }
      }
      Mapped.push_back(ND);
    }
  }
}

void ConfusableIdentifierCheck::registerMatchers(
    ast_matchers::MatchFinder *Finder) {
  Finder->addMatcher(ast_matchers::namedDecl().bind("nameddecl"), this);
}

} // namespace misc
} // namespace tidy
} // namespace clang

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
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/ConvertUTF.h"

namespace {
// Preprocessed version of
// https://www.unicode.org/Public/security/latest/confusables.txt
//
// This contains a sorted array of { UTF32 codepoint; UTF32 values[N];}
#include "Confusables.inc"
} // namespace

namespace clang::tidy::misc {

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

static llvm::SmallString<64U> skeleton(StringRef Name) {
  using namespace llvm;
  SmallString<64U> Skeleton;
  Skeleton.reserve(1U + Name.size());

  const char *Curr = Name.data();
  const char *End = Curr + Name.size();
  while (Curr < End) {

    const char *Prev = Curr;
    UTF32 CodePoint = 0;
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

namespace {
struct Entry {
  const NamedDecl *ND;
  bool FromDerivedClass;
};
}

using DeclsWithinContextMap =
    llvm::DenseMap<const DeclContext *, llvm::SmallVector<Entry, 1>>;

static bool addToContext(DeclsWithinContextMap &DeclsWithinContext,
                         const DeclContext *DC, Entry E) {
  auto &Decls = DeclsWithinContext[DC];
  if (!Decls.empty() &&
      Decls.back().ND->getIdentifier() == E.ND->getIdentifier()) {
    // Already have a declaration with this identifier in this context. Don't
    // track another one. This means that if an outer name is confusable with an
    // inner name, we'll only diagnose the outer name once, pointing at the
    // first inner declaration with that name.
    if (Decls.back().FromDerivedClass && !E.FromDerivedClass) {
      // Prefer the declaration that's not from the derived class, because that
      // conflicts with more declarations.
      Decls.back() = E;
      return true;
    }
    return false;
  }
  Decls.push_back(E);
  return true;
}

static void addToEnclosingContexts(DeclsWithinContextMap &DeclsWithinContext,
                                   const DeclContext *DC, const NamedDecl *ND) {
  while (DC) {
    DC = DC->getNonTransparentContext()->getPrimaryContext();
    if (!addToContext(DeclsWithinContext, DC, {ND, false}))
      return;

    if (const auto *RD = dyn_cast<CXXRecordDecl>(DC)) {
      RD = RD->getDefinition();
      if (RD) {
        RD->forallBases([&](const CXXRecordDecl *Base) {
          addToContext(DeclsWithinContext, Base, {ND, true});
          return true;
        });
      }
    }

    DC = DC->getParent();
  }
}

void ConfusableIdentifierCheck::check(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  const auto *ND = Result.Nodes.getNodeAs<NamedDecl>("nameddecl");
  if (!ND)
    return;

  const IdentifierInfo *NDII = ND->getIdentifier();
  if (!NDII)
    return;

  StringRef NDName = NDII->getName();
  if (NDName.empty())
    return;

  NameToDecls[NDII].push_back(ND);
}

void ConfusableIdentifierCheck::onEndOfTranslationUnit() {
  llvm::StringMap<llvm::SmallVector<const IdentifierInfo*, 1>> SkeletonToNames;
  // Compute the skeleton for each identifier.
  for (auto &[Ident, Decls] : NameToDecls) {
    SkeletonToNames[skeleton(Ident->getName())].push_back(Ident);
  }

  // Visit each skeleton with more than one identifier.
  for (auto &[Skel, Idents] : SkeletonToNames) {
    if (Idents.size() < 2) {
      continue;
    }

    // Find the declaration contexts that transitively contain each identifier.
    DeclsWithinContextMap DeclsWithinContext;
    for (const IdentifierInfo *II : Idents) {
      for (const NamedDecl *ND : NameToDecls[II]) {
        addToEnclosingContexts(DeclsWithinContext, ND->getDeclContext(), ND);
      }
    }

    // Check to see if any declaration is declared in a context that
    // transitively contains another declaration with a different identifier but
    // the same skeleton.
    for (const IdentifierInfo *II : Idents) {
      for (const NamedDecl *OuterND : NameToDecls[II]) {
        const DeclContext *OuterDC = OuterND->getDeclContext()
                                         ->getNonTransparentContext()
                                         ->getPrimaryContext();
        for (Entry Inner : DeclsWithinContext[OuterDC]) {
          // Don't complain if the identifiers are the same.
          if (OuterND->getIdentifier() == Inner.ND->getIdentifier())
            continue;

          // Don't complain about a derived-class name shadowing a base class
          // private member.
          if (OuterND->getAccess() == AS_private && Inner.FromDerivedClass)
            continue;

          // If the declarations are in the same context, only diagnose the
          // later one.
          if (OuterDC->Equals(
                  Inner.ND->getDeclContext()->getNonTransparentContext()) &&
              Inner.ND->getASTContext()
                  .getSourceManager()
                  .isBeforeInTranslationUnit(Inner.ND->getLocation(),
                                             OuterND->getLocation()))
            continue;

          diag(Inner.ND->getLocation(), "%0 is confusable with %1")
              << Inner.ND << OuterND;
          diag(OuterND->getLocation(), "other declaration found here",
                DiagnosticIDs::Note);
        }
      }
    }
  }

  NameToDecls.clear();
}

void ConfusableIdentifierCheck::registerMatchers(
    ast_matchers::MatchFinder *Finder) {
  Finder->addMatcher(ast_matchers::namedDecl().bind("nameddecl"), this);
}

} // namespace clang::tidy::misc

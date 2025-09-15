//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ConfusableIdentifierCheck.h"

#include "clang/ASTMatchers/ASTMatchers.h"
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
    auto *Where = llvm::lower_bound(ConfusableEntries, CodePoint,
                                    [](decltype(ConfusableEntries[0]) X,
                                       UTF32 Y) { return X.codepoint < Y; });
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
  const Decl *Parent;
  bool FromDerivedClass;
};
} // namespace

// Map from a context to the declarations in that context with the current
// skeleton. At most one entry per distinct identifier is tracked. The
// context is usually a `DeclContext`, but can also be a template declaration
// that has no corresponding context, such as an alias template or variable
// template.
using DeclsWithinContextMap =
    llvm::DenseMap<const Decl *, llvm::SmallVector<Entry, 1>>;

static bool addToContext(DeclsWithinContextMap &DeclsWithinContext,
                         const Decl *Context, Entry E) {
  auto &Decls = DeclsWithinContext[Context];
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
                                   const Decl *Parent, const NamedDecl *ND) {
  const Decl *Outer = Parent;
  while (Outer) {
    if (const auto *NS = dyn_cast<NamespaceDecl>(Outer))
      Outer = NS->getCanonicalDecl();

    if (!addToContext(DeclsWithinContext, Outer, {ND, Parent, false}))
      return;

    if (const auto *RD = dyn_cast<CXXRecordDecl>(Outer)) {
      RD = RD->getDefinition();
      if (RD) {
        RD->forallBases([&](const CXXRecordDecl *Base) {
          addToContext(DeclsWithinContext, Base, {ND, Parent, true});
          return true;
        });
      }
    }

    auto *OuterDC = Outer->getDeclContext();
    if (!OuterDC)
      break;
    Outer = cast_or_null<Decl>(OuterDC->getNonTransparentContext());
  }
}

void ConfusableIdentifierCheck::check(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  const auto *ND = Result.Nodes.getNodeAs<NamedDecl>("nameddecl");
  if (!ND)
    return;

  addDeclToCheck(ND,
                 cast<Decl>(ND->getDeclContext()->getNonTransparentContext()));

  // Associate template parameters with this declaration of this template.
  if (const auto *TD = dyn_cast<TemplateDecl>(ND)) {
    for (const NamedDecl *Param : *TD->getTemplateParameters())
      addDeclToCheck(Param, TD->getTemplatedDecl());
  }

  // Associate function parameters with this declaration of this function.
  if (const auto *FD = dyn_cast<FunctionDecl>(ND)) {
    for (const NamedDecl *Param : FD->parameters())
      addDeclToCheck(Param, ND);
  }
}

void ConfusableIdentifierCheck::addDeclToCheck(const NamedDecl *ND,
                                               const Decl *Parent) {
  if (!ND || !Parent)
    return;

  const IdentifierInfo *NDII = ND->getIdentifier();
  if (!NDII)
    return;

  StringRef NDName = NDII->getName();
  if (NDName.empty())
    return;

  NameToDecls[NDII].push_back({ND, Parent});
}

void ConfusableIdentifierCheck::onEndOfTranslationUnit() {
  llvm::StringMap<llvm::SmallVector<const IdentifierInfo *, 1>> SkeletonToNames;
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
      for (auto [ND, Parent] : NameToDecls[II]) {
        addToEnclosingContexts(DeclsWithinContext, Parent, ND);
      }
    }

    // Check to see if any declaration is declared in a context that
    // transitively contains another declaration with a different identifier but
    // the same skeleton.
    for (const IdentifierInfo *II : Idents) {
      for (auto [OuterND, OuterParent] : NameToDecls[II]) {
        for (Entry Inner : DeclsWithinContext[OuterParent]) {
          // Don't complain if the identifiers are the same.
          if (OuterND->getIdentifier() == Inner.ND->getIdentifier())
            continue;

          // Don't complain about a derived-class name shadowing a base class
          // private member.
          if (OuterND->getAccess() == AS_private && Inner.FromDerivedClass)
            continue;

          // If the declarations are in the same context, only diagnose the
          // later one.
          if (OuterParent == Inner.Parent &&
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
  // Parameter declarations sometimes use the translation unit or some outer
  // enclosing context as their `DeclContext`, instead of their parent, so
  // we handle them specially in `check`.
  auto AnyParamDecl = ast_matchers::anyOf(
      ast_matchers::parmVarDecl(), ast_matchers::templateTypeParmDecl(),
      ast_matchers::nonTypeTemplateParmDecl(),
      ast_matchers::templateTemplateParmDecl());
  Finder->addMatcher(ast_matchers::namedDecl(ast_matchers::unless(AnyParamDecl))
                         .bind("nameddecl"),
                     this);
}

} // namespace clang::tidy::misc

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

static bool mayShadowImpl(const DeclContext *DC0, const DeclContext *DC1) {
  return DC0 && DC0 == DC1;
}

static bool mayShadowImpl(const NamedDecl *ND0, const NamedDecl *ND1) {
  return isa<TemplateTypeParmDecl>(ND0) || isa<TemplateTypeParmDecl>(ND1);
}

static bool isMemberOf(const ConfusableIdentifierCheck::ContextInfo *DC0,
                       const ConfusableIdentifierCheck::ContextInfo *DC1) {
  return llvm::is_contained(DC1->Bases, DC0->PrimaryContext);
}

static bool enclosesContext(const ConfusableIdentifierCheck::ContextInfo *DC0,
                            const ConfusableIdentifierCheck::ContextInfo *DC1) {
  if (DC0->PrimaryContext == DC1->PrimaryContext)
    return true;

  return llvm::is_contained(DC0->PrimaryContexts, DC1->PrimaryContext) ||
         llvm::is_contained(DC1->PrimaryContexts, DC0->PrimaryContext);
}

static bool mayShadow(const NamedDecl *ND0,
                      const ConfusableIdentifierCheck::ContextInfo *DC0,
                      const NamedDecl *ND1,
                      const ConfusableIdentifierCheck::ContextInfo *DC1) {

  if (!DC0->Bases.empty() && !DC1->Bases.empty()) {
    // if any of the declaration is a non-private member of the other
    // declaration, it's shadowed by the former

    if (ND1->getAccess() != AS_private && isMemberOf(DC1, DC0))
      return true;

    if (ND0->getAccess() != AS_private && isMemberOf(DC0, DC1))
      return true;
  }

  if (!mayShadowImpl(DC0->NonTransparentContext, DC1->NonTransparentContext) &&
      !mayShadowImpl(ND0, ND1))
    return false;

  return enclosesContext(DC0, DC1);
}

const ConfusableIdentifierCheck::ContextInfo *
ConfusableIdentifierCheck::getContextInfo(const DeclContext *DC) {
  const DeclContext *PrimaryContext = DC->getPrimaryContext();
  auto It = ContextInfos.find(PrimaryContext);
  if (It != ContextInfos.end())
    return &It->second;

  ContextInfo &Info = ContextInfos[PrimaryContext];
  Info.PrimaryContext = PrimaryContext;
  Info.NonTransparentContext = PrimaryContext;

  while (Info.NonTransparentContext->isTransparentContext()) {
    Info.NonTransparentContext = Info.NonTransparentContext->getParent();
    if (!Info.NonTransparentContext)
      break;
  }

  if (Info.NonTransparentContext)
    Info.NonTransparentContext =
        Info.NonTransparentContext->getPrimaryContext();

  while (DC) {
    if (!isa<LinkageSpecDecl>(DC) && !isa<ExportDecl>(DC))
      Info.PrimaryContexts.push_back(DC->getPrimaryContext());
    DC = DC->getParent();
  }

  if (const auto *RD = dyn_cast<CXXRecordDecl>(PrimaryContext)) {
    RD = RD->getDefinition();
    if (RD) {
      Info.Bases.push_back(RD);
      RD->forallBases([&](const CXXRecordDecl *Base) {
        Info.Bases.push_back(Base);
        return false;
      });
    }
  }

  return &Info;
}

void ConfusableIdentifierCheck::check(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  const auto *ND = Result.Nodes.getNodeAs<NamedDecl>("nameddecl");
  if (!ND)
    return;

  IdentifierInfo *NDII = ND->getIdentifier();
  if (!NDII)
    return;

  StringRef NDName = NDII->getName();
  if (NDName.empty())
    return;

  const ContextInfo *Info = getContextInfo(ND->getDeclContext());

  llvm::SmallVector<Entry> &Mapped = Mapper[skeleton(NDName)];
  for (const Entry &E : Mapped) {
    if (!mayShadow(ND, Info, E.Declaration, E.Info))
      continue;

    const IdentifierInfo *ONDII = E.Declaration->getIdentifier();
    StringRef ONDName = ONDII->getName();
    if (ONDName == NDName)
      continue;

    diag(ND->getLocation(), "%0 is confusable with %1") << ND << E.Declaration;
    diag(E.Declaration->getLocation(), "other declaration found here",
         DiagnosticIDs::Note);
  }

  Mapped.push_back({ND, Info});
}

void ConfusableIdentifierCheck::onEndOfTranslationUnit() {
  Mapper.clear();
  ContextInfos.clear();
}

void ConfusableIdentifierCheck::registerMatchers(
    ast_matchers::MatchFinder *Finder) {
  Finder->addMatcher(ast_matchers::namedDecl().bind("nameddecl"), this);
}

} // namespace clang::tidy::misc

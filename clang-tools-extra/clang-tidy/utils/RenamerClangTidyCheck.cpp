//===--- RenamerClangTidyCheck.cpp - clang-tidy ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RenamerClangTidyCheck.h"
#include "ASTUtils.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"

#define DEBUG_TYPE "clang-tidy"

using namespace clang::ast_matchers;

namespace llvm {

/// Specialisation of DenseMapInfo to allow NamingCheckId objects in DenseMaps
template <>
struct DenseMapInfo<clang::tidy::RenamerClangTidyCheck::NamingCheckId> {
  using NamingCheckId = clang::tidy::RenamerClangTidyCheck::NamingCheckId;

  static inline NamingCheckId getEmptyKey() {
    return NamingCheckId(
        clang::SourceLocation::getFromRawEncoding(static_cast<unsigned>(-1)),
        "EMPTY");
  }

  static inline NamingCheckId getTombstoneKey() {
    return NamingCheckId(
        clang::SourceLocation::getFromRawEncoding(static_cast<unsigned>(-2)),
        "TOMBSTONE");
  }

  static unsigned getHashValue(NamingCheckId Val) {
    assert(Val != getEmptyKey() && "Cannot hash the empty key!");
    assert(Val != getTombstoneKey() && "Cannot hash the tombstone key!");

    std::hash<NamingCheckId::second_type> SecondHash;
    return Val.first.getRawEncoding() + SecondHash(Val.second);
  }

  static bool isEqual(const NamingCheckId &LHS, const NamingCheckId &RHS) {
    if (RHS == getEmptyKey())
      return LHS == getEmptyKey();
    if (RHS == getTombstoneKey())
      return LHS == getTombstoneKey();
    return LHS == RHS;
  }
};

} // namespace llvm

namespace clang {
namespace tidy {

namespace {

/// Callback supplies macros to RenamerClangTidyCheck::checkMacro
class RenamerClangTidyCheckPPCallbacks : public PPCallbacks {
public:
  RenamerClangTidyCheckPPCallbacks(Preprocessor *PP,
                                   RenamerClangTidyCheck *Check)
      : PP(PP), Check(Check) {}

  /// MacroDefined calls checkMacro for macros in the main file
  void MacroDefined(const Token &MacroNameTok,
                    const MacroDirective *MD) override {
    Check->checkMacro(PP->getSourceManager(), MacroNameTok, MD->getMacroInfo());
  }

  /// MacroExpands calls expandMacro for macros in the main file
  void MacroExpands(const Token &MacroNameTok, const MacroDefinition &MD,
                    SourceRange /*Range*/,
                    const MacroArgs * /*Args*/) override {
    Check->expandMacro(MacroNameTok, MD.getMacroInfo());
  }

private:
  Preprocessor *PP;
  RenamerClangTidyCheck *Check;
};

} // namespace

RenamerClangTidyCheck::RenamerClangTidyCheck(StringRef CheckName,
                                             ClangTidyContext *Context)
    : ClangTidyCheck(CheckName, Context) {}
RenamerClangTidyCheck::~RenamerClangTidyCheck() = default;

void RenamerClangTidyCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(namedDecl().bind("decl"), this);
  Finder->addMatcher(usingDecl().bind("using"), this);
  Finder->addMatcher(declRefExpr().bind("declRef"), this);
  Finder->addMatcher(cxxConstructorDecl(unless(isImplicit())).bind("classRef"),
                     this);
  Finder->addMatcher(cxxDestructorDecl(unless(isImplicit())).bind("classRef"),
                     this);
  Finder->addMatcher(typeLoc().bind("typeLoc"), this);
  Finder->addMatcher(nestedNameSpecifierLoc().bind("nestedNameLoc"), this);
  Finder->addMatcher(
      functionDecl(unless(cxxMethodDecl(isImplicit())),
                   hasBody(forEachDescendant(memberExpr().bind("memberExpr")))),
      this);
  Finder->addMatcher(
      cxxConstructorDecl(
          unless(isImplicit()),
          forEachConstructorInitializer(
              allOf(isWritten(), withInitializer(forEachDescendant(
                                     memberExpr().bind("memberExpr")))))),
      this);
  Finder->addMatcher(fieldDecl(hasInClassInitializer(
                         forEachDescendant(memberExpr().bind("memberExpr")))),
                     this);
}

void RenamerClangTidyCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  ModuleExpanderPP->addPPCallbacks(
      std::make_unique<RenamerClangTidyCheckPPCallbacks>(ModuleExpanderPP,
                                                         this));
}

static void addUsage(RenamerClangTidyCheck::NamingCheckFailureMap &Failures,
                     const RenamerClangTidyCheck::NamingCheckId &Decl,
                     SourceRange Range, SourceManager *SourceMgr = nullptr) {
  // Do nothing if the provided range is invalid.
  if (Range.getBegin().isInvalid() || Range.getEnd().isInvalid())
    return;

  // If we have a source manager, use it to convert to the spelling location for
  // performing the fix. This is necessary because macros can map the same
  // spelling location to different source locations, and we only want to fix
  // the token once, before it is expanded by the macro.
  SourceLocation FixLocation = Range.getBegin();
  if (SourceMgr)
    FixLocation = SourceMgr->getSpellingLoc(FixLocation);
  if (FixLocation.isInvalid())
    return;

  // Try to insert the identifier location in the Usages map, and bail out if it
  // is already in there
  RenamerClangTidyCheck::NamingCheckFailure &Failure = Failures[Decl];
  if (!Failure.RawUsageLocs.insert(FixLocation.getRawEncoding()).second)
    return;

  if (!Failure.ShouldFix())
    return;

  if (!utils::rangeCanBeFixed(Range, SourceMgr))
    Failure.FixStatus = RenamerClangTidyCheck::ShouldFixStatus::InsideMacro;
}

/// Convenience method when the usage to be added is a NamedDecl
static void addUsage(RenamerClangTidyCheck::NamingCheckFailureMap &Failures,
                     const NamedDecl *Decl, SourceRange Range,
                     SourceManager *SourceMgr = nullptr) {
  return addUsage(Failures,
                  RenamerClangTidyCheck::NamingCheckId(Decl->getLocation(),
                                                       Decl->getNameAsString()),
                  Range, SourceMgr);
}

void RenamerClangTidyCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *Decl =
          Result.Nodes.getNodeAs<CXXConstructorDecl>("classRef")) {

    addUsage(NamingCheckFailures, Decl->getParent(),
             Decl->getNameInfo().getSourceRange());

    for (const auto *Init : Decl->inits()) {
      if (!Init->isWritten() || Init->isInClassMemberInitializer())
        continue;
      if (const FieldDecl *FD = Init->getAnyMember())
        addUsage(NamingCheckFailures, FD,
                 SourceRange(Init->getMemberLocation()));
      // Note: delegating constructors and base class initializers are handled
      // via the "typeLoc" matcher.
    }
    return;
  }

  if (const auto *Decl =
          Result.Nodes.getNodeAs<CXXDestructorDecl>("classRef")) {

    SourceRange Range = Decl->getNameInfo().getSourceRange();
    if (Range.getBegin().isInvalid())
      return;
    // The first token that will be found is the ~ (or the equivalent trigraph),
    // we want instead to replace the next token, that will be the identifier.
    Range.setBegin(CharSourceRange::getTokenRange(Range).getEnd());

    addUsage(NamingCheckFailures, Decl->getParent(), Range);
    return;
  }

  if (const auto *Loc = Result.Nodes.getNodeAs<TypeLoc>("typeLoc")) {
    NamedDecl *Decl = nullptr;
    if (const auto &Ref = Loc->getAs<TagTypeLoc>())
      Decl = Ref.getDecl();
    else if (const auto &Ref = Loc->getAs<InjectedClassNameTypeLoc>())
      Decl = Ref.getDecl();
    else if (const auto &Ref = Loc->getAs<UnresolvedUsingTypeLoc>())
      Decl = Ref.getDecl();
    else if (const auto &Ref = Loc->getAs<TemplateTypeParmTypeLoc>())
      Decl = Ref.getDecl();
    // further TypeLocs handled below

    if (Decl) {
      addUsage(NamingCheckFailures, Decl, Loc->getSourceRange());
      return;
    }

    if (const auto &Ref = Loc->getAs<TemplateSpecializationTypeLoc>()) {
      const TemplateDecl *Decl =
          Ref.getTypePtr()->getTemplateName().getAsTemplateDecl();

      SourceRange Range(Ref.getTemplateNameLoc(), Ref.getTemplateNameLoc());
      if (const auto *ClassDecl = dyn_cast<TemplateDecl>(Decl)) {
        if (const NamedDecl *TemplDecl = ClassDecl->getTemplatedDecl())
          addUsage(NamingCheckFailures, TemplDecl, Range);
        return;
      }
    }

    if (const auto &Ref =
            Loc->getAs<DependentTemplateSpecializationTypeLoc>()) {
      if (const TagDecl *Decl = Ref.getTypePtr()->getAsTagDecl())
        addUsage(NamingCheckFailures, Decl, Loc->getSourceRange());
      return;
    }
  }

  if (const auto *Loc =
          Result.Nodes.getNodeAs<NestedNameSpecifierLoc>("nestedNameLoc")) {
    if (const NestedNameSpecifier *Spec = Loc->getNestedNameSpecifier()) {
      if (const NamespaceDecl *Decl = Spec->getAsNamespace()) {
        addUsage(NamingCheckFailures, Decl, Loc->getLocalSourceRange());
        return;
      }
    }
  }

  if (const auto *Decl = Result.Nodes.getNodeAs<UsingDecl>("using")) {
    for (const auto *Shadow : Decl->shadows())
      addUsage(NamingCheckFailures, Shadow->getTargetDecl(),
               Decl->getNameInfo().getSourceRange());
    return;
  }

  if (const auto *DeclRef = Result.Nodes.getNodeAs<DeclRefExpr>("declRef")) {
    SourceRange Range = DeclRef->getNameInfo().getSourceRange();
    addUsage(NamingCheckFailures, DeclRef->getDecl(), Range,
             Result.SourceManager);
    return;
  }

  if (const auto *MemberRef =
          Result.Nodes.getNodeAs<MemberExpr>("memberExpr")) {
    SourceRange Range = MemberRef->getMemberNameInfo().getSourceRange();
    addUsage(NamingCheckFailures, MemberRef->getMemberDecl(), Range,
             Result.SourceManager);
    return;
  }

  if (const auto *Decl = Result.Nodes.getNodeAs<NamedDecl>("decl")) {
    if (!Decl->getIdentifier() || Decl->getName().empty() || Decl->isImplicit())
      return;

    // Fix type aliases in value declarations.
    if (const auto *Value = Result.Nodes.getNodeAs<ValueDecl>("decl")) {
      if (const Type *TypePtr = Value->getType().getTypePtrOrNull()) {
        if (const auto *Typedef = TypePtr->getAs<TypedefType>())
          addUsage(NamingCheckFailures, Typedef->getDecl(),
                   Value->getSourceRange());
      }
    }

    // Fix type aliases in function declarations.
    if (const auto *Value = Result.Nodes.getNodeAs<FunctionDecl>("decl")) {
      if (const auto *Typedef =
              Value->getReturnType().getTypePtr()->getAs<TypedefType>())
        addUsage(NamingCheckFailures, Typedef->getDecl(),
                 Value->getSourceRange());
      for (unsigned i = 0; i < Value->getNumParams(); ++i) {
        if (const TypedefType *Typedef = Value->parameters()[i]
                                             ->getType()
                                             .getTypePtr()
                                             ->getAs<TypedefType>())
          addUsage(NamingCheckFailures, Typedef->getDecl(),
                   Value->getSourceRange());
      }
    }

    // Ignore ClassTemplateSpecializationDecl which are creating duplicate
    // replacements with CXXRecordDecl.
    if (isa<ClassTemplateSpecializationDecl>(Decl))
      return;

    Optional<FailureInfo> MaybeFailure =
        GetDeclFailureInfo(Decl, *Result.SourceManager);
    if (!MaybeFailure)
      return;
    FailureInfo &Info = *MaybeFailure;
    NamingCheckFailure &Failure = NamingCheckFailures[NamingCheckId(
        Decl->getLocation(), Decl->getNameAsString())];
    SourceRange Range =
        DeclarationNameInfo(Decl->getDeclName(), Decl->getLocation())
            .getSourceRange();

    const IdentifierTable &Idents = Decl->getASTContext().Idents;
    auto CheckNewIdentifier = Idents.find(Info.Fixup);
    if (CheckNewIdentifier != Idents.end()) {
      const IdentifierInfo *Ident = CheckNewIdentifier->second;
      if (Ident->isKeyword(getLangOpts()))
        Failure.FixStatus = ShouldFixStatus::ConflictsWithKeyword;
      else if (Ident->hasMacroDefinition())
        Failure.FixStatus = ShouldFixStatus::ConflictsWithMacroDefinition;
    }

    Failure.Info = std::move(Info);
    addUsage(NamingCheckFailures, Decl, Range);
  }
}

void RenamerClangTidyCheck::checkMacro(SourceManager &SourceMgr,
                                       const Token &MacroNameTok,
                                       const MacroInfo *MI) {
  Optional<FailureInfo> MaybeFailure =
      GetMacroFailureInfo(MacroNameTok, SourceMgr);
  if (!MaybeFailure)
    return;
  FailureInfo &Info = *MaybeFailure;
  StringRef Name = MacroNameTok.getIdentifierInfo()->getName();
  NamingCheckId ID(MI->getDefinitionLoc(), Name);
  NamingCheckFailure &Failure = NamingCheckFailures[ID];
  SourceRange Range(MacroNameTok.getLocation(), MacroNameTok.getEndLoc());

  Failure.Info = std::move(Info);
  addUsage(NamingCheckFailures, ID, Range);
}

void RenamerClangTidyCheck::expandMacro(const Token &MacroNameTok,
                                        const MacroInfo *MI) {
  StringRef Name = MacroNameTok.getIdentifierInfo()->getName();
  NamingCheckId ID(MI->getDefinitionLoc(), Name);

  auto Failure = NamingCheckFailures.find(ID);
  if (Failure == NamingCheckFailures.end())
    return;

  SourceRange Range(MacroNameTok.getLocation(), MacroNameTok.getEndLoc());
  addUsage(NamingCheckFailures, ID, Range);
}

static std::string
getDiagnosticSuffix(const RenamerClangTidyCheck::ShouldFixStatus FixStatus,
                    const std::string &Fixup) {
  if (Fixup.empty())
    return "; cannot be fixed automatically";
  if (FixStatus == RenamerClangTidyCheck::ShouldFixStatus::ShouldFix)
    return {};
  if (FixStatus >=
      RenamerClangTidyCheck::ShouldFixStatus::IgnoreFailureThreshold)
    return {};
  if (FixStatus == RenamerClangTidyCheck::ShouldFixStatus::ConflictsWithKeyword)
    return "; cannot be fixed because '" + Fixup +
           "' would conflict with a keyword";
  if (FixStatus ==
      RenamerClangTidyCheck::ShouldFixStatus::ConflictsWithMacroDefinition)
    return "; cannot be fixed because '" + Fixup +
           "' would conflict with a macro definition";

  llvm_unreachable("invalid ShouldFixStatus");
}

void RenamerClangTidyCheck::onEndOfTranslationUnit() {
  for (const auto &Pair : NamingCheckFailures) {
    const NamingCheckId &Decl = Pair.first;
    const NamingCheckFailure &Failure = Pair.second;

    if (Failure.Info.KindName.empty())
      continue;

    if (Failure.ShouldNotify()) {
      auto DiagInfo = GetDiagInfo(Decl, Failure);
      auto Diag = diag(Decl.first,
                       DiagInfo.Text + getDiagnosticSuffix(Failure.FixStatus,
                                                           Failure.Info.Fixup));
      DiagInfo.ApplyArgs(Diag);

      if (Failure.ShouldFix()) {
        for (const auto &Loc : Failure.RawUsageLocs) {
          // We assume that the identifier name is made of one token only. This
          // is always the case as we ignore usages in macros that could build
          // identifier names by combining multiple tokens.
          //
          // For destructors, we already take care of it by remembering the
          // location of the start of the identifier and not the start of the
          // tilde.
          //
          // Other multi-token identifiers, such as operators are not checked at
          // all.
          Diag << FixItHint::CreateReplacement(
              SourceRange(SourceLocation::getFromRawEncoding(Loc)),
              Failure.Info.Fixup);
        }
      }
    }
  }
}

} // namespace tidy
} // namespace clang

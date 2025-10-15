//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RenamerClangTidyCheck.h"
#include "ASTUtils.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/PointerIntPair.h"
#include <optional>

#define DEBUG_TYPE "clang-tidy"

using namespace clang::ast_matchers;

namespace llvm {

/// Specialization of DenseMapInfo to allow NamingCheckId objects in DenseMaps
template <>
struct DenseMapInfo<clang::tidy::RenamerClangTidyCheck::NamingCheckId> {
  using NamingCheckId = clang::tidy::RenamerClangTidyCheck::NamingCheckId;

  static NamingCheckId getEmptyKey() {
    return {DenseMapInfo<clang::SourceLocation>::getEmptyKey(), "EMPTY"};
  }

  static NamingCheckId getTombstoneKey() {
    return {DenseMapInfo<clang::SourceLocation>::getTombstoneKey(),
            "TOMBSTONE"};
  }

  static unsigned getHashValue(NamingCheckId Val) {
    assert(Val != getEmptyKey() && "Cannot hash the empty key!");
    assert(Val != getTombstoneKey() && "Cannot hash the tombstone key!");

    return DenseMapInfo<clang::SourceLocation>::getHashValue(Val.first) +
           DenseMapInfo<StringRef>::getHashValue(Val.second);
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

namespace clang::tidy {

namespace {

class NameLookup {
  llvm::PointerIntPair<const NamedDecl *, 1, bool> Data;

public:
  explicit NameLookup(const NamedDecl *ND) : Data(ND, false) {}
  explicit NameLookup(std::nullopt_t) : Data(nullptr, true) {}
  explicit NameLookup(std::nullptr_t) : Data(nullptr, false) {}
  NameLookup() : NameLookup(nullptr) {}

  bool hasMultipleResolutions() const { return Data.getInt(); }
  const NamedDecl *getDecl() const {
    assert(!hasMultipleResolutions() && "Found multiple decls");
    return Data.getPointer();
  }
  operator bool() const { return !hasMultipleResolutions(); }
  const NamedDecl *operator*() const { return getDecl(); }
};

} // namespace

static const NamedDecl *findDecl(const RecordDecl &RecDecl,
                                 StringRef DeclName) {
  for (const Decl *D : RecDecl.decls()) {
    if (const auto *ND = dyn_cast<NamedDecl>(D)) {
      if (ND->getDeclName().isIdentifier() && ND->getName() == DeclName)
        return ND;
    }
  }
  return nullptr;
}

/// Returns the function that \p Method is overridding. If There are none or
/// multiple overrides it returns nullptr. If the overridden function itself is
/// overridding then it will recurse up to find the first decl of the function.
static const CXXMethodDecl *getOverrideMethod(const CXXMethodDecl *Method) {
  if (Method->size_overridden_methods() != 1)
    return nullptr;

  while (true) {
    Method = *Method->begin_overridden_methods();
    assert(Method && "Overridden method shouldn't be null");
    unsigned NumOverrides = Method->size_overridden_methods();
    if (NumOverrides == 0)
      return Method;
    if (NumOverrides > 1)
      return nullptr;
  }
}

static bool hasNoName(const NamedDecl *Decl) {
  return !Decl->getIdentifier() || Decl->getName().empty();
}

static const NamedDecl *getFailureForNamedDecl(const NamedDecl *ND) {
  const auto *Canonical = cast<NamedDecl>(ND->getCanonicalDecl());
  if (Canonical != ND)
    return Canonical;

  if (const auto *Method = dyn_cast<CXXMethodDecl>(ND)) {
    if (const CXXMethodDecl *Overridden = getOverrideMethod(Method))
      Canonical = cast<NamedDecl>(Overridden->getCanonicalDecl());
    else if (const FunctionTemplateDecl *Primary = Method->getPrimaryTemplate())
      if (const FunctionDecl *TemplatedDecl = Primary->getTemplatedDecl())
        Canonical = cast<NamedDecl>(TemplatedDecl->getCanonicalDecl());

    if (Canonical != ND)
      return Canonical;
  }

  return ND;
}

/// Returns a decl matching the \p DeclName in \p Parent or one of its base
/// classes. If \p AggressiveTemplateLookup is `true` then it will check
/// template dependent base classes as well.
/// If a matching decl is found in multiple base classes then it will return a
/// flag indicating the multiple resolutions.
static NameLookup findDeclInBases(const CXXRecordDecl &Parent,
                                  StringRef DeclName,
                                  bool AggressiveTemplateLookup) {
  if (!Parent.hasDefinition())
    return NameLookup(nullptr);
  if (const NamedDecl *InClassRef = findDecl(Parent, DeclName))
    return NameLookup(InClassRef);
  const NamedDecl *Found = nullptr;

  for (CXXBaseSpecifier Base : Parent.bases()) {
    const auto *Record = Base.getType()->getAsCXXRecordDecl();
    if (!Record && AggressiveTemplateLookup) {
      if (const auto *TST =
              Base.getType()->getAs<TemplateSpecializationType>()) {
        if (const auto *TD = llvm::dyn_cast_or_null<ClassTemplateDecl>(
                TST->getTemplateName().getAsTemplateDecl()))
          Record = TD->getTemplatedDecl();
      }
    }
    if (!Record)
      continue;
    if (auto Search =
            findDeclInBases(*Record, DeclName, AggressiveTemplateLookup)) {
      if (*Search) {
        if (Found)
          return NameLookup(
              std::nullopt); // Multiple decls found in different base classes.
        Found = *Search;
        continue;
      }
    } else
      return NameLookup(std::nullopt); // Propagate multiple resolution back up.
  }
  return NameLookup(Found); // If nullptr, decl wasn't found.
}

namespace {

/// Callback supplies macros to RenamerClangTidyCheck::checkMacro
class RenamerClangTidyCheckPPCallbacks : public PPCallbacks {
public:
  RenamerClangTidyCheckPPCallbacks(const SourceManager &SM,
                                   RenamerClangTidyCheck *Check)
      : SM(SM), Check(Check) {}

  /// MacroDefined calls checkMacro for macros in the main file
  void MacroDefined(const Token &MacroNameTok,
                    const MacroDirective *MD) override {
    const MacroInfo *Info = MD->getMacroInfo();
    if (Info->isBuiltinMacro())
      return;
    if (SM.isWrittenInBuiltinFile(MacroNameTok.getLocation()))
      return;
    if (SM.isWrittenInCommandLineFile(MacroNameTok.getLocation()))
      return;
    if (SM.isInSystemHeader(MacroNameTok.getLocation()))
      return;
    Check->checkMacro(MacroNameTok, Info, SM);
  }

  /// MacroExpands calls expandMacro for macros in the main file
  void MacroExpands(const Token &MacroNameTok, const MacroDefinition &MD,
                    SourceRange /*Range*/,
                    const MacroArgs * /*Args*/) override {
    Check->expandMacro(MacroNameTok, MD.getMacroInfo(), SM);
  }

private:
  const SourceManager &SM;
  RenamerClangTidyCheck *Check;
};

class RenamerClangTidyVisitor
    : public RecursiveASTVisitor<RenamerClangTidyVisitor> {
public:
  RenamerClangTidyVisitor(RenamerClangTidyCheck *Check, const SourceManager &SM,
                          bool AggressiveDependentMemberLookup)
      : Check(Check), SM(SM),
        AggressiveDependentMemberLookup(AggressiveDependentMemberLookup) {}

  bool shouldVisitTemplateInstantiations() const { return true; }

  bool shouldVisitImplicitCode() const { return false; }

  bool VisitCXXConstructorDecl(CXXConstructorDecl *Decl) {
    if (Decl->isImplicit())
      return true;
    Check->addUsage(Decl->getParent(), Decl->getNameInfo().getSourceRange(),
                    SM);

    for (const auto *Init : Decl->inits()) {
      if (!Init->isWritten() || Init->isInClassMemberInitializer())
        continue;
      if (const FieldDecl *FD = Init->getAnyMember())
        Check->addUsage(FD, SourceRange(Init->getMemberLocation()), SM);
      // Note: delegating constructors and base class initializers are handled
      // via the "typeLoc" matcher.
    }

    return true;
  }

  bool VisitCXXDestructorDecl(CXXDestructorDecl *Decl) {
    if (Decl->isImplicit())
      return true;
    SourceRange Range = Decl->getNameInfo().getSourceRange();
    if (Range.getBegin().isInvalid())
      return true;

    // The first token that will be found is the ~ (or the equivalent trigraph),
    // we want instead to replace the next token, that will be the identifier.
    Range.setBegin(CharSourceRange::getTokenRange(Range).getEnd());
    Check->addUsage(Decl->getParent(), Range, SM);
    return true;
  }

  bool VisitUsingDecl(UsingDecl *Decl) {
    for (const auto *Shadow : Decl->shadows())
      Check->addUsage(Shadow->getTargetDecl(),
                      Decl->getNameInfo().getSourceRange(), SM);
    return true;
  }

  bool VisitUsingDirectiveDecl(UsingDirectiveDecl *Decl) {
    Check->addUsage(Decl->getNominatedNamespaceAsWritten(),
                    Decl->getIdentLocation(), SM);
    return true;
  }

  bool VisitNamedDecl(NamedDecl *Decl) {
    SourceRange UsageRange =
        DeclarationNameInfo(Decl->getDeclName(), Decl->getLocation())
            .getSourceRange();
    Check->addUsage(Decl, UsageRange, SM);
    return true;
  }

  bool VisitDeclRefExpr(DeclRefExpr *DeclRef) {
    SourceRange Range = DeclRef->getNameInfo().getSourceRange();
    Check->addUsage(DeclRef->getDecl(), Range, SM);
    return true;
  }

  bool TraverseNestedNameSpecifierLoc(NestedNameSpecifierLoc Loc) {
    if (NestedNameSpecifier Spec = Loc.getNestedNameSpecifier();
        Spec.getKind() == NestedNameSpecifier::Kind::Namespace) {
      if (const auto *Decl =
              dyn_cast<NamespaceDecl>(Spec.getAsNamespaceAndPrefix().Namespace))
        Check->addUsage(Decl, Loc.getLocalSourceRange(), SM);
    }

    using Base = RecursiveASTVisitor<RenamerClangTidyVisitor>;
    return Base::TraverseNestedNameSpecifierLoc(Loc);
  }

  bool VisitMemberExpr(MemberExpr *MemberRef) {
    SourceRange Range = MemberRef->getMemberNameInfo().getSourceRange();
    Check->addUsage(MemberRef->getMemberDecl(), Range, SM);
    return true;
  }

  bool
  VisitCXXDependentScopeMemberExpr(CXXDependentScopeMemberExpr *DepMemberRef) {
    QualType BaseType = DepMemberRef->isArrow()
                            ? DepMemberRef->getBaseType()->getPointeeType()
                            : DepMemberRef->getBaseType();
    if (BaseType.isNull())
      return true;
    const CXXRecordDecl *Base = BaseType.getTypePtr()->getAsCXXRecordDecl();
    if (!Base)
      return true;
    DeclarationName DeclName = DepMemberRef->getMemberNameInfo().getName();
    if (!DeclName.isIdentifier())
      return true;
    StringRef DependentName = DeclName.getAsIdentifierInfo()->getName();

    if (NameLookup Resolved = findDeclInBases(
            *Base, DependentName, AggressiveDependentMemberLookup)) {
      if (*Resolved)
        Check->addUsage(*Resolved,
                        DepMemberRef->getMemberNameInfo().getSourceRange(), SM);
    }

    return true;
  }

  bool VisitTypedefTypeLoc(const TypedefTypeLoc &Loc) {
    Check->addUsage(Loc.getDecl(), Loc.getNameLoc(), SM);
    return true;
  }

  bool VisitTagTypeLoc(const TagTypeLoc &Loc) {
    Check->addUsage(Loc.getDecl(), Loc.getNameLoc(), SM);
    return true;
  }

  bool VisitUnresolvedUsingTypeLoc(const UnresolvedUsingTypeLoc &Loc) {
    Check->addUsage(Loc.getDecl(), Loc.getNameLoc(), SM);
    return true;
  }

  bool VisitTemplateTypeParmTypeLoc(const TemplateTypeParmTypeLoc &Loc) {
    Check->addUsage(Loc.getDecl(), Loc.getNameLoc(), SM);
    return true;
  }

  bool
  VisitTemplateSpecializationTypeLoc(const TemplateSpecializationTypeLoc &Loc) {
    const TemplateDecl *Decl =
        Loc.getTypePtr()->getTemplateName().getAsTemplateDecl(
            /*IgnoreDeduced=*/true);
    if (!Decl)
      return true;

    if (const auto *ClassDecl = dyn_cast<TemplateDecl>(Decl))
      if (const NamedDecl *TemplDecl = ClassDecl->getTemplatedDecl())
        Check->addUsage(TemplDecl, Loc.getTemplateNameLoc(), SM);

    return true;
  }

  bool VisitDesignatedInitExpr(DesignatedInitExpr *Expr) {
    for (const DesignatedInitExpr::Designator &D : Expr->designators()) {
      if (!D.isFieldDesignator())
        continue;
      const FieldDecl *FD = D.getFieldDecl();
      if (!FD)
        continue;
      const IdentifierInfo *II = FD->getIdentifier();
      if (!II)
        continue;
      SourceRange FixLocation{D.getFieldLoc(), D.getFieldLoc()};
      Check->addUsage(FD, FixLocation, SM);
    }

    return true;
  }

private:
  RenamerClangTidyCheck *Check;
  const SourceManager &SM;
  const bool AggressiveDependentMemberLookup;
};

} // namespace

RenamerClangTidyCheck::RenamerClangTidyCheck(StringRef CheckName,
                                             ClangTidyContext *Context)
    : ClangTidyCheck(CheckName, Context),
      AggressiveDependentMemberLookup(
          Options.get("AggressiveDependentMemberLookup", false)) {}
RenamerClangTidyCheck::~RenamerClangTidyCheck() = default;

void RenamerClangTidyCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "AggressiveDependentMemberLookup",
                AggressiveDependentMemberLookup);
}

void RenamerClangTidyCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(translationUnitDecl(), this);
}

void RenamerClangTidyCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  ModuleExpanderPP->addPPCallbacks(
      std::make_unique<RenamerClangTidyCheckPPCallbacks>(SM, this));
}

std::pair<RenamerClangTidyCheck::NamingCheckFailureMap::iterator, bool>
RenamerClangTidyCheck::addUsage(
    const RenamerClangTidyCheck::NamingCheckId &FailureId,
    SourceRange UsageRange, const SourceManager &SourceMgr) {
  // Do nothing if the provided range is invalid.
  if (UsageRange.isInvalid())
    return {NamingCheckFailures.end(), false};

  // Get the spelling location for performing the fix. This is necessary because
  // macros can map the same spelling location to different source locations,
  // and we only want to fix the token once, before it is expanded by the macro.
  SourceLocation FixLocation = UsageRange.getBegin();
  FixLocation = SourceMgr.getSpellingLoc(FixLocation);
  if (FixLocation.isInvalid())
    return {NamingCheckFailures.end(), false};

  // Skip if in system system header
  if (SourceMgr.isInSystemHeader(FixLocation))
    return {NamingCheckFailures.end(), false};

  auto EmplaceResult = NamingCheckFailures.try_emplace(FailureId);
  NamingCheckFailure &Failure = EmplaceResult.first->second;

  // Try to insert the identifier location in the Usages map, and bail out if it
  // is already in there
  if (!Failure.RawUsageLocs.insert(FixLocation).second)
    return EmplaceResult;

  if (Failure.FixStatus != RenamerClangTidyCheck::ShouldFixStatus::ShouldFix)
    return EmplaceResult;

  if (SourceMgr.isWrittenInScratchSpace(FixLocation))
    Failure.FixStatus = RenamerClangTidyCheck::ShouldFixStatus::InsideMacro;

  if (!utils::rangeCanBeFixed(UsageRange, &SourceMgr))
    Failure.FixStatus = RenamerClangTidyCheck::ShouldFixStatus::InsideMacro;

  return EmplaceResult;
}

void RenamerClangTidyCheck::addUsage(const NamedDecl *Decl,
                                     SourceRange UsageRange,
                                     const SourceManager &SourceMgr) {
  if (SourceMgr.isInSystemHeader(Decl->getLocation()))
    return;

  if (hasNoName(Decl))
    return;

  // Ignore ClassTemplateSpecializationDecl which are creating duplicate
  // replacements with CXXRecordDecl.
  if (isa<ClassTemplateSpecializationDecl>(Decl))
    return;

  // We don't want to create a failure for every NamedDecl we find. Ideally
  // there is just one NamedDecl in every group of "related" NamedDecls that
  // becomes the failure. This NamedDecl and all of its related NamedDecls
  // become usages. E.g. Since NamedDecls are Redeclarable, only the canonical
  // NamedDecl becomes the failure and all redeclarations become usages.
  const NamedDecl *FailureDecl = getFailureForNamedDecl(Decl);

  std::optional<FailureInfo> MaybeFailure =
      getDeclFailureInfo(FailureDecl, SourceMgr);
  if (!MaybeFailure)
    return;

  NamingCheckId FailureId(FailureDecl->getLocation(), FailureDecl->getName());

  auto [FailureIter, NewFailure] = addUsage(FailureId, UsageRange, SourceMgr);

  if (FailureIter == NamingCheckFailures.end()) {
    // Nothing to do if the usage wasn't accepted.
    return;
  }
  if (!NewFailure) {
    // FailureInfo has already been provided.
    return;
  }

  // Update the stored failure with info regarding the FailureDecl.
  NamingCheckFailure &Failure = FailureIter->second;
  Failure.Info = std::move(*MaybeFailure);

  // Don't overwritte the failure status if it was already set.
  if (!Failure.shouldFix()) {
    return;
  }
  const IdentifierTable &Idents = FailureDecl->getASTContext().Idents;
  auto CheckNewIdentifier = Idents.find(Failure.Info.Fixup);
  if (CheckNewIdentifier != Idents.end()) {
    const IdentifierInfo *Ident = CheckNewIdentifier->second;
    if (Ident->isKeyword(getLangOpts()))
      Failure.FixStatus = ShouldFixStatus::ConflictsWithKeyword;
    else if (Ident->hasMacroDefinition())
      Failure.FixStatus = ShouldFixStatus::ConflictsWithMacroDefinition;
  } else if (!isValidAsciiIdentifier(Failure.Info.Fixup)) {
    Failure.FixStatus = ShouldFixStatus::FixInvalidIdentifier;
  }
}

void RenamerClangTidyCheck::check(const MatchFinder::MatchResult &Result) {
  if (!Result.SourceManager) {
    // In principle SourceManager is not null but going only by the definition
    // of MatchResult it must be handled. Cannot rename anything without a
    // SourceManager.
    return;
  }
  RenamerClangTidyVisitor Visitor(this, *Result.SourceManager,
                                  AggressiveDependentMemberLookup);
  Visitor.TraverseAST(*Result.Context);
}

void RenamerClangTidyCheck::checkMacro(const Token &MacroNameTok,
                                       const MacroInfo *MI,
                                       const SourceManager &SourceMgr) {
  std::optional<FailureInfo> MaybeFailure =
      getMacroFailureInfo(MacroNameTok, SourceMgr);
  if (!MaybeFailure)
    return;
  FailureInfo &Info = *MaybeFailure;
  StringRef Name = MacroNameTok.getIdentifierInfo()->getName();
  NamingCheckId ID(MI->getDefinitionLoc(), Name);
  NamingCheckFailure &Failure = NamingCheckFailures[ID];
  SourceRange Range(MacroNameTok.getLocation(), MacroNameTok.getEndLoc());

  if (!isValidAsciiIdentifier(Info.Fixup))
    Failure.FixStatus = ShouldFixStatus::FixInvalidIdentifier;

  Failure.Info = std::move(Info);
  addUsage(ID, Range, SourceMgr);
}

void RenamerClangTidyCheck::expandMacro(const Token &MacroNameTok,
                                        const MacroInfo *MI,
                                        const SourceManager &SourceMgr) {
  StringRef Name = MacroNameTok.getIdentifierInfo()->getName();
  NamingCheckId ID(MI->getDefinitionLoc(), Name);

  auto Failure = NamingCheckFailures.find(ID);
  if (Failure == NamingCheckFailures.end())
    return;

  SourceRange Range(MacroNameTok.getLocation(), MacroNameTok.getEndLoc());
  addUsage(ID, Range, SourceMgr);
}

static std::string
getDiagnosticSuffix(const RenamerClangTidyCheck::ShouldFixStatus FixStatus,
                    const std::string &Fixup) {
  if (Fixup.empty() ||
      FixStatus == RenamerClangTidyCheck::ShouldFixStatus::FixInvalidIdentifier)
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

    if (Failure.shouldNotify()) {
      auto DiagInfo = getDiagInfo(Decl, Failure);
      auto Diag = diag(Decl.first,
                       DiagInfo.Text + getDiagnosticSuffix(Failure.FixStatus,
                                                           Failure.Info.Fixup));
      DiagInfo.ApplyArgs(Diag);

      if (Failure.shouldFix()) {
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
          Diag << FixItHint::CreateReplacement(SourceRange(Loc),
                                               Failure.Info.Fixup);
        }
      }
    }
  }
}

} // namespace clang::tidy

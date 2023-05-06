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

  static inline NamingCheckId getEmptyKey() {
    return NamingCheckId(DenseMapInfo<clang::SourceLocation>::getEmptyKey(),
                         "EMPTY");
  }

  static inline NamingCheckId getTombstoneKey() {
    return NamingCheckId(DenseMapInfo<clang::SourceLocation>::getTombstoneKey(),
                         "TOMBSTONE");
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
      if (ND->getDeclName().isIdentifier() && ND->getName().equals(DeclName))
        return ND;
    }
  }
  return nullptr;
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
    Check->checkMacro(SM, MacroNameTok, Info);
  }

  /// MacroExpands calls expandMacro for macros in the main file
  void MacroExpands(const Token &MacroNameTok, const MacroDefinition &MD,
                    SourceRange /*Range*/,
                    const MacroArgs * /*Args*/) override {
    Check->expandMacro(MacroNameTok, MD.getMacroInfo());
  }

private:
  const SourceManager &SM;
  RenamerClangTidyCheck *Check;
};

class RenamerClangTidyVisitor
    : public RecursiveASTVisitor<RenamerClangTidyVisitor> {
public:
  RenamerClangTidyVisitor(RenamerClangTidyCheck *Check, const SourceManager *SM,
                          bool AggressiveDependentMemberLookup)
      : Check(Check), SM(SM),
        AggressiveDependentMemberLookup(AggressiveDependentMemberLookup) {}

  static bool hasNoName(const NamedDecl *Decl) {
    return !Decl->getIdentifier() || Decl->getName().empty();
  }

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
    if (hasNoName(Decl))
      return true;

    const auto *Canonical = cast<NamedDecl>(Decl->getCanonicalDecl());
    if (Canonical != Decl) {
      Check->addUsage(Canonical, Decl->getLocation(), SM);
      return true;
    }

    // Fix type aliases in value declarations.
    if (const auto *Value = dyn_cast<ValueDecl>(Decl)) {
      if (const Type *TypePtr = Value->getType().getTypePtrOrNull()) {
        if (const auto *Typedef = TypePtr->getAs<TypedefType>())
          Check->addUsage(Typedef->getDecl(), Value->getSourceRange(), SM);
      }
    }

    // Fix type aliases in function declarations.
    if (const auto *Value = dyn_cast<FunctionDecl>(Decl)) {
      if (const auto *Typedef =
              Value->getReturnType().getTypePtr()->getAs<TypedefType>())
        Check->addUsage(Typedef->getDecl(), Value->getSourceRange(), SM);
      for (const ParmVarDecl *Param : Value->parameters()) {
        if (const TypedefType *Typedef =
                Param->getType().getTypePtr()->getAs<TypedefType>())
          Check->addUsage(Typedef->getDecl(), Value->getSourceRange(), SM);
      }
    }

    // Fix overridden methods
    if (const auto *Method = dyn_cast<CXXMethodDecl>(Decl)) {
      if (const CXXMethodDecl *Overridden = getOverrideMethod(Method)) {
        Check->addUsage(Overridden, Method->getLocation());
        return true; // Don't try to add the actual decl as a Failure.
      }
    }

    // Ignore ClassTemplateSpecializationDecl which are creating duplicate
    // replacements with CXXRecordDecl.
    if (isa<ClassTemplateSpecializationDecl>(Decl))
      return true;

    Check->checkNamedDecl(Decl, *SM);
    return true;
  }

  bool VisitDeclRefExpr(DeclRefExpr *DeclRef) {
    SourceRange Range = DeclRef->getNameInfo().getSourceRange();
    Check->addUsage(DeclRef->getDecl(), Range, SM);
    return true;
  }

  bool TraverseNestedNameSpecifierLoc(NestedNameSpecifierLoc Loc) {
    if (const NestedNameSpecifier *Spec = Loc.getNestedNameSpecifier()) {
      if (const NamespaceDecl *Decl = Spec->getAsNamespace())
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

  bool VisitTagTypeLoc(const TagTypeLoc &Loc) {
    Check->addUsage(Loc.getDecl(), Loc.getSourceRange(), SM);
    return true;
  }

  bool VisitInjectedClassNameTypeLoc(const InjectedClassNameTypeLoc &Loc) {
    Check->addUsage(Loc.getDecl(), Loc.getSourceRange(), SM);
    return true;
  }

  bool VisitUnresolvedUsingTypeLoc(const UnresolvedUsingTypeLoc &Loc) {
    Check->addUsage(Loc.getDecl(), Loc.getSourceRange(), SM);
    return true;
  }

  bool VisitTemplateTypeParmTypeLoc(const TemplateTypeParmTypeLoc &Loc) {
    Check->addUsage(Loc.getDecl(), Loc.getSourceRange(), SM);
    return true;
  }

  bool
  VisitTemplateSpecializationTypeLoc(const TemplateSpecializationTypeLoc &Loc) {
    const TemplateDecl *Decl =
        Loc.getTypePtr()->getTemplateName().getAsTemplateDecl();

    SourceRange Range(Loc.getTemplateNameLoc(), Loc.getTemplateNameLoc());
    if (const auto *ClassDecl = dyn_cast<TemplateDecl>(Decl)) {
      if (const NamedDecl *TemplDecl = ClassDecl->getTemplatedDecl())
        Check->addUsage(TemplDecl, Range, SM);
    }

    return true;
  }

  bool VisitDependentTemplateSpecializationTypeLoc(
      const DependentTemplateSpecializationTypeLoc &Loc) {
    if (const TagDecl *Decl = Loc.getTypePtr()->getAsTagDecl())
      Check->addUsage(Decl, Loc.getSourceRange(), SM);

    return true;
  }

private:
  RenamerClangTidyCheck *Check;
  const SourceManager *SM;
  const bool AggressiveDependentMemberLookup;
};

} // namespace

RenamerClangTidyCheck::RenamerClangTidyCheck(StringRef CheckName,
                                             ClangTidyContext *Context)
    : ClangTidyCheck(CheckName, Context),
      AggressiveDependentMemberLookup(
          Options.getLocalOrGlobal("AggressiveDependentMemberLookup", false)) {}
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

void RenamerClangTidyCheck::addUsage(
    const RenamerClangTidyCheck::NamingCheckId &Decl, SourceRange Range,
    const SourceManager *SourceMgr) {
  // Do nothing if the provided range is invalid.
  if (Range.isInvalid())
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
  RenamerClangTidyCheck::NamingCheckFailure &Failure =
      NamingCheckFailures[Decl];
  if (!Failure.RawUsageLocs.insert(FixLocation).second)
    return;

  if (!Failure.shouldFix())
    return;

  if (SourceMgr && SourceMgr->isWrittenInScratchSpace(FixLocation))
    Failure.FixStatus = RenamerClangTidyCheck::ShouldFixStatus::InsideMacro;

  if (!utils::rangeCanBeFixed(Range, SourceMgr))
    Failure.FixStatus = RenamerClangTidyCheck::ShouldFixStatus::InsideMacro;
}

void RenamerClangTidyCheck::addUsage(const NamedDecl *Decl, SourceRange Range,
                                     const SourceManager *SourceMgr) {
  if (const auto *Method = dyn_cast<CXXMethodDecl>(Decl)) {
    if (const CXXMethodDecl *Overridden = getOverrideMethod(Method))
      Decl = Overridden;
  }
  Decl = cast<NamedDecl>(Decl->getCanonicalDecl());
  return addUsage(RenamerClangTidyCheck::NamingCheckId(Decl->getLocation(),
                                                       Decl->getName()),
                  Range, SourceMgr);
}

void RenamerClangTidyCheck::checkNamedDecl(const NamedDecl *Decl,
                                           const SourceManager &SourceMgr) {
  std::optional<FailureInfo> MaybeFailure = getDeclFailureInfo(Decl, SourceMgr);
  if (!MaybeFailure)
    return;

  FailureInfo &Info = *MaybeFailure;
  NamingCheckFailure &Failure =
      NamingCheckFailures[NamingCheckId(Decl->getLocation(), Decl->getName())];
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
  } else if (!isValidAsciiIdentifier(Info.Fixup)) {
    Failure.FixStatus = ShouldFixStatus::FixInvalidIdentifier;
  }

  Failure.Info = std::move(Info);
  addUsage(Decl, Range);
}

void RenamerClangTidyCheck::check(const MatchFinder::MatchResult &Result) {
  RenamerClangTidyVisitor Visitor(this, Result.SourceManager,
                                  AggressiveDependentMemberLookup);
  Visitor.TraverseAST(*Result.Context);
}

void RenamerClangTidyCheck::checkMacro(const SourceManager &SourceMgr,
                                       const Token &MacroNameTok,
                                       const MacroInfo *MI) {
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
  addUsage(ID, Range);
}

void RenamerClangTidyCheck::expandMacro(const Token &MacroNameTok,
                                        const MacroInfo *MI) {
  StringRef Name = MacroNameTok.getIdentifierInfo()->getName();
  NamingCheckId ID(MI->getDefinitionLoc(), Name);

  auto Failure = NamingCheckFailures.find(ID);
  if (Failure == NamingCheckFailures.end())
    return;

  SourceRange Range(MacroNameTok.getLocation(), MacroNameTok.getEndLoc());
  addUsage(ID, Range);
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

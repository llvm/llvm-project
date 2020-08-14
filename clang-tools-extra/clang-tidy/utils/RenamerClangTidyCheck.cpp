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
#include "llvm/ADT/PointerIntPair.h"

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
    if (MD->getMacroInfo()->isBuiltinMacro())
      return;
    if (PP->getSourceManager().isWrittenInBuiltinFile(
            MacroNameTok.getLocation()))
      return;
    if (PP->getSourceManager().isWrittenInCommandLineFile(
            MacroNameTok.getLocation()))
      return;
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
    : ClangTidyCheck(CheckName, Context),
      AggressiveDependentMemberLookup(
          Options.getLocalOrGlobal("AggressiveDependentMemberLookup", false)) {}
RenamerClangTidyCheck::~RenamerClangTidyCheck() = default;

void RenamerClangTidyCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "AggressiveDependentMemberLookup",
                AggressiveDependentMemberLookup);
}

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
  auto MemberRestrictions =
      unless(forFunction(anyOf(isDefaulted(), isImplicit())));
  Finder->addMatcher(memberExpr(MemberRestrictions).bind("memberExpr"), this);
  Finder->addMatcher(
      cxxDependentScopeMemberExpr(MemberRestrictions).bind("depMemberExpr"),
      this);
}

void RenamerClangTidyCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  ModuleExpanderPP->addPPCallbacks(
      std::make_unique<RenamerClangTidyCheckPPCallbacks>(ModuleExpanderPP,
                                                         this));
}

void RenamerClangTidyCheck::addUsage(
    const RenamerClangTidyCheck::NamingCheckId &Decl, SourceRange Range,
    SourceManager *SourceMgr) {
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

  if (!Failure.RawUsageLocs.insert(FixLocation.getRawEncoding()).second)
    return;

  if (!Failure.ShouldFix())
    return;

  if (SourceMgr && SourceMgr->isWrittenInScratchSpace(FixLocation))
    Failure.FixStatus = RenamerClangTidyCheck::ShouldFixStatus::InsideMacro;

  if (!utils::rangeCanBeFixed(Range, SourceMgr))
    Failure.FixStatus = RenamerClangTidyCheck::ShouldFixStatus::InsideMacro;
}

void RenamerClangTidyCheck::addUsage(const NamedDecl *Decl, SourceRange Range,
                                     SourceManager *SourceMgr) {
  Decl = cast<NamedDecl>(Decl->getCanonicalDecl());
  return addUsage(RenamerClangTidyCheck::NamingCheckId(Decl->getLocation(),
                                                       Decl->getNameAsString()),
                  Range, SourceMgr);
}

const NamedDecl *findDecl(const RecordDecl &RecDecl, StringRef DeclName) {
  for (const Decl *D : RecDecl.decls()) {
    if (const auto *ND = dyn_cast<NamedDecl>(D)) {
      if (ND->getDeclName().isIdentifier() && ND->getName().equals(DeclName))
        return ND;
    }
  }
  return nullptr;
}

namespace {
class NameLookup {
  llvm::PointerIntPair<const NamedDecl *, 1, bool> Data;

public:
  explicit NameLookup(const NamedDecl *ND) : Data(ND, false) {}
  explicit NameLookup(llvm::NoneType) : Data(nullptr, true) {}
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

/// Returns a decl matching the \p DeclName in \p Parent or one of its base
/// classes. If \p AggressiveTemplateLookup is `true` then it will check
/// template dependent base classes as well.
/// If a matching decl is found in multiple base classes then it will return a
/// flag indicating the multiple resolutions.
NameLookup findDeclInBases(const CXXRecordDecl &Parent, StringRef DeclName,
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
              llvm::None); // Multiple decls found in different base classes.
        Found = *Search;
        continue;
      }
    } else
      return NameLookup(llvm::None); // Propagate multiple resolution back up.
  }
  return NameLookup(Found); // If nullptr, decl wasnt found.
}

void RenamerClangTidyCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *Decl =
          Result.Nodes.getNodeAs<CXXConstructorDecl>("classRef")) {

    addUsage(Decl->getParent(), Decl->getNameInfo().getSourceRange(),
             Result.SourceManager);

    for (const auto *Init : Decl->inits()) {
      if (!Init->isWritten() || Init->isInClassMemberInitializer())
        continue;
      if (const FieldDecl *FD = Init->getAnyMember())
        addUsage(FD, SourceRange(Init->getMemberLocation()),
                 Result.SourceManager);
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

    addUsage(Decl->getParent(), Range, Result.SourceManager);
    return;
  }

  if (const auto *Loc = Result.Nodes.getNodeAs<TypeLoc>("typeLoc")) {
    UnqualTypeLoc Unqual = Loc->getUnqualifiedLoc();
    NamedDecl *Decl = nullptr;
    if (const auto &Ref = Unqual.getAs<TagTypeLoc>())
      Decl = Ref.getDecl();
    else if (const auto &Ref = Unqual.getAs<InjectedClassNameTypeLoc>())
      Decl = Ref.getDecl();
    else if (const auto &Ref = Unqual.getAs<UnresolvedUsingTypeLoc>())
      Decl = Ref.getDecl();
    else if (const auto &Ref = Unqual.getAs<TemplateTypeParmTypeLoc>())
      Decl = Ref.getDecl();
    // further TypeLocs handled below

    if (Decl) {
      addUsage(Decl, Loc->getSourceRange(), Result.SourceManager);
      return;
    }

    if (const auto &Ref = Loc->getAs<TemplateSpecializationTypeLoc>()) {
      const TemplateDecl *Decl =
          Ref.getTypePtr()->getTemplateName().getAsTemplateDecl();

      SourceRange Range(Ref.getTemplateNameLoc(), Ref.getTemplateNameLoc());
      if (const auto *ClassDecl = dyn_cast<TemplateDecl>(Decl)) {
        if (const NamedDecl *TemplDecl = ClassDecl->getTemplatedDecl())
          addUsage(TemplDecl, Range, Result.SourceManager);
        return;
      }
    }

    if (const auto &Ref =
            Loc->getAs<DependentTemplateSpecializationTypeLoc>()) {
      if (const TagDecl *Decl = Ref.getTypePtr()->getAsTagDecl())
        addUsage(Decl, Loc->getSourceRange(), Result.SourceManager);
      return;
    }
  }

  if (const auto *Loc =
          Result.Nodes.getNodeAs<NestedNameSpecifierLoc>("nestedNameLoc")) {
    if (const NestedNameSpecifier *Spec = Loc->getNestedNameSpecifier()) {
      if (const NamespaceDecl *Decl = Spec->getAsNamespace()) {
        addUsage(Decl, Loc->getLocalSourceRange(), Result.SourceManager);
        return;
      }
    }
  }

  if (const auto *Decl = Result.Nodes.getNodeAs<UsingDecl>("using")) {
    for (const auto *Shadow : Decl->shadows())
      addUsage(Shadow->getTargetDecl(), Decl->getNameInfo().getSourceRange(),
               Result.SourceManager);
    return;
  }

  if (const auto *DeclRef = Result.Nodes.getNodeAs<DeclRefExpr>("declRef")) {
    SourceRange Range = DeclRef->getNameInfo().getSourceRange();
    addUsage(DeclRef->getDecl(), Range, Result.SourceManager);
    return;
  }

  if (const auto *MemberRef =
          Result.Nodes.getNodeAs<MemberExpr>("memberExpr")) {
    SourceRange Range = MemberRef->getMemberNameInfo().getSourceRange();
    addUsage(MemberRef->getMemberDecl(), Range, Result.SourceManager);
    return;
  }

  if (const auto *DepMemberRef =
          Result.Nodes.getNodeAs<CXXDependentScopeMemberExpr>(
              "depMemberExpr")) {
    QualType BaseType = DepMemberRef->isArrow()
                            ? DepMemberRef->getBaseType()->getPointeeType()
                            : DepMemberRef->getBaseType();
    if (BaseType.isNull())
      return;
    const CXXRecordDecl *Base = BaseType.getTypePtr()->getAsCXXRecordDecl();
    if (!Base)
      return;
    DeclarationName DeclName = DepMemberRef->getMemberNameInfo().getName();
    if (!DeclName.isIdentifier())
      return;
    StringRef DependentName = DeclName.getAsIdentifierInfo()->getName();

    if (NameLookup Resolved = findDeclInBases(
            *Base, DependentName, AggressiveDependentMemberLookup)) {
      if (*Resolved)
        addUsage(*Resolved, DepMemberRef->getMemberNameInfo().getSourceRange(),
                 Result.SourceManager);
    }
    return;
  }

  if (const auto *Decl = Result.Nodes.getNodeAs<NamedDecl>("decl")) {
    // Fix using namespace declarations.
    if (const auto *UsingNS = dyn_cast<UsingDirectiveDecl>(Decl))
      addUsage(UsingNS->getNominatedNamespaceAsWritten(),
               UsingNS->getIdentLocation(), Result.SourceManager);

    if (!Decl->getIdentifier() || Decl->getName().empty() || Decl->isImplicit())
      return;

    const auto *Canonical = cast<NamedDecl>(Decl->getCanonicalDecl());
    if (Canonical != Decl) {
      addUsage(Canonical, Decl->getLocation(), Result.SourceManager);
      return;
    }

    // Fix type aliases in value declarations.
    if (const auto *Value = Result.Nodes.getNodeAs<ValueDecl>("decl")) {
      if (const Type *TypePtr = Value->getType().getTypePtrOrNull()) {
        if (const auto *Typedef = TypePtr->getAs<TypedefType>())
          addUsage(Typedef->getDecl(), Value->getSourceRange(),
                   Result.SourceManager);
      }
    }

    // Fix type aliases in function declarations.
    if (const auto *Value = Result.Nodes.getNodeAs<FunctionDecl>("decl")) {
      if (const auto *Typedef =
              Value->getReturnType().getTypePtr()->getAs<TypedefType>())
        addUsage(Typedef->getDecl(), Value->getSourceRange(),
                 Result.SourceManager);
      for (const ParmVarDecl *Param : Value->parameters()) {
        if (const TypedefType *Typedef =
                Param->getType().getTypePtr()->getAs<TypedefType>())
          addUsage(Typedef->getDecl(), Value->getSourceRange(),
                   Result.SourceManager);
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
    addUsage(Decl, Range);
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
  NamingCheckId ID(MI->getDefinitionLoc(), std::string(Name));
  NamingCheckFailure &Failure = NamingCheckFailures[ID];
  SourceRange Range(MacroNameTok.getLocation(), MacroNameTok.getEndLoc());

  Failure.Info = std::move(Info);
  addUsage(ID, Range);
}

void RenamerClangTidyCheck::expandMacro(const Token &MacroNameTok,
                                        const MacroInfo *MI) {
  StringRef Name = MacroNameTok.getIdentifierInfo()->getName();
  NamingCheckId ID(MI->getDefinitionLoc(), std::string(Name));

  auto Failure = NamingCheckFailures.find(ID);
  if (Failure == NamingCheckFailures.end())
    return;

  SourceRange Range(MacroNameTok.getLocation(), MacroNameTok.getEndLoc());
  addUsage(ID, Range);
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

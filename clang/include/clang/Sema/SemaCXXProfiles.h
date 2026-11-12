//===----- SemaCXXProfiles.h ---- C++ profiles (P3589) ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares semantic analysis for C++ profiles (P3589): the
/// profiles::enforce, profiles::suppress, profiles::require and
/// profiles::exempt attributes, and the checking of profile rules.
///
/// Rule checks run where Sema builds the relevant construct, but violations
/// are only recorded there. They are reported once the outermost enclosing
/// function (or the translation unit) is complete, so that
/// [[profiles::suppress]] attributes on enclosing statements, which are
/// attached after the statement has been analyzed, can be taken into account.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SEMACXXPROFILES_H
#define LLVM_CLANG_SEMA_SEMACXXPROFILES_H

#include "clang/AST/ASTFwd.h"
#include "clang/AST/Type.h"
#include "clang/Basic/CXXProfiles.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/PartialDiagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/SemaBase.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>
#include <string>
#include <vector>

namespace clang {
class Attr;
class ExplicitCastExpr;
class ImportDecl;
class Module;
class ParsedAttr;
class ProfilesSuppressAttr;

class SemaCXXProfiles : public SemaBase {
public:
  SemaCXXProfiles(Sema &S);

  /// Process the profiles enforced on the command line.
  void ActOnStartOfTranslationUnit();

  /// Report the remaining profile violations and record the profiles
  /// enforced by the current module unit, if any.
  void ActOnEndOfTranslationUnit();

  /// Report the profile violations recorded so far. Called when the
  /// outermost function scope is popped and at the end of the translation
  /// unit.
  void flushPendingViolations();

  /// \name Attributes
  /// @{
  void handleEnforceAttr(Decl *D, const ParsedAttr &AL);
  void handleSuppressAttr(Decl *D, const ParsedAttr &AL);
  Attr *handleSuppressStmtAttr(Stmt *St, const ParsedAttr &AL);
  void handleExemptAttr(Decl *D, const ParsedAttr &AL);
  /// Diagnose a [[profiles::require]] attribute that does not appertain to a
  /// module import declaration.
  void handleMisplacedRequireAttr(const ParsedAttr &AL);
  /// @}

  /// \name Modules
  /// @{

  /// Called when a module declaration has been processed. \p Interface is the
  /// primary module interface implicitly imported by a module implementation
  /// unit, if any.
  void ActOnModuleDecl(Module *Mod, Module *Interface,
                       SourceLocation ModuleLoc);

  /// Process a [[profiles::enforce]] attribute on the module declaration.
  void ActOnModuleDeclEnforceAttr(const ParsedAttr &AL);

  /// Check a [[profiles::require]] attribute on a module import declaration.
  void ActOnModuleImportRequireAttr(ImportDecl *Import, const ParsedAttr &AL);
  /// @}

  /// \name Rule checks
  /// @{

  /// Check an explicit cast against the rules of std::casting.
  /// \p CastAwayQualifiers holds the qualifiers the cast casts away
  /// ([expr.const.cast]), if it casts away constness.
  void checkExplicitCast(const ExplicitCastExpr *E,
                         std::optional<Qualifiers> CastAwayQualifiers);
  /// @}

private:
  struct EnforcedProfile {
    /// The profile, if it is known to Clang.
    std::optional<CXXProfileID> ID;
    /// The profile designator as written (canonical spelling).
    std::string Designator;
    /// The location of the designator; invalid if the profile is enforced on
    /// the command line or inherited from the primary module interface.
    SourceLocation Loc;
  };

  struct SuppressionSite {
    CXXProfileID Profile;
    /// The suppressed rule, or std::nullopt if every rule of the profile is
    /// suppressed.
    std::optional<CXXProfileRuleID> Rule;
    /// The declaration or statement the suppression appertains to.
    llvm::PointerUnion<const Decl *, const Stmt *> Node;
  };

  struct ExemptHeader {
    CXXProfileID Profile;
    std::string Header;
  };

  struct PendingViolation {
    CXXProfileRuleID Rule;
    /// The declaration context in which the violation was found; used to find
    /// the enforcing unit and suppressions of templates from other modules.
    const Decl *Owner;
    PartialDiagnosticAt Diag;
    SmallVector<PartialDiagnosticAt, 1> Notes;
  };

  /// Record the profile named by \p Designator as enforced in \p List.
  /// Returns false if it was already enforced (or conflicts with an existing
  /// enforcement, which is diagnosed if \p Diagnose is true).
  bool addEnforcedProfile(StringRef Designator, SourceLocation Loc,
                          SmallVectorImpl<EnforcedProfile> &List,
                          bool Diagnose = true);

  /// Returns the first non-empty declaration in the translation unit that
  /// precedes \p D, if any.
  const Decl *findPrecedingNonEmptyDecl(const Decl *D) const;

  bool isEnforcedInCurrentUnit(CXXProfileID Profile, SourceLocation Loc) const;

  /// Returns true if the rules of \p Profile apply to code at \p Loc within
  /// the declaration context \p Owner.
  bool isEnforced(CXXProfileID Profile, SourceLocation Loc,
                  const Decl *Owner) const;

  /// Returns true if code at \p Loc is exempt from \p Profile: it is in a
  /// system header or in a header named by a [[profiles::exempt]] attribute.
  bool isExempt(CXXProfileID Profile, SourceLocation Loc) const;

  /// Returns true if the rules of \p Profile should be checked for code at
  /// \p Loc in the current context.
  bool shouldCheck(CXXProfileID Profile, SourceLocation Loc) const;

  bool isSuppressed(CXXProfileRuleID Rule, SourceLocation Loc,
                    const Decl *Owner);

  std::optional<SuppressionSite> makeSuppressionSite(
      const ProfilesSuppressAttr *A,
      llvm::PointerUnion<const Decl *, const Stmt *> Node) const;

  /// Returns the suppressions that appertain to a template from an AST file
  /// whose instantiation is being checked in this translation unit.
  ArrayRef<SuppressionSite> getImportedSuppressionSites(const Decl *Pattern);

  void addViolation(CXXProfileRuleID Rule, PartialDiagnosticAt Diag,
                    SmallVector<PartialDiagnosticAt, 1> Notes = {});

  /// Record the profiles enforced by the current module unit on its Module so
  /// that they are serialized.
  void updateModuleRecord();

  bool Initialized = false;
  SmallVector<EnforcedProfile, 2> UnitProfiles;
  SmallVector<EnforcedProfile, 2> ModuleProfiles;
  SmallVector<CXXProfileID, 1> PreviewProfiles;
  SourceLocation ModuleDeclLoc;
  Module *UnitModule = nullptr;
  Module *PrimaryInterface = nullptr;

  SmallVector<SuppressionSite, 4> SuppressionSites;
  llvm::DenseMap<const Decl *, SmallVector<SuppressionSite, 2>>
      ImportedSuppressionSites;
  SmallVector<ExemptHeader, 1> ExemptHeaders;

  SmallVector<PendingViolation, 4> PendingViolations;
  /// The (rule, location) pairs already reported, so that a violation in a
  /// template is reported once rather than once per instantiation.
  llvm::DenseSet<std::pair<unsigned, SourceLocation::UIntTy>>
      ReportedViolations;
};

} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMACXXPROFILES_H

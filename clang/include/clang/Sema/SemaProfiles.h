//===----- SemaProfiles.h --- C++ profiles framework ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares semantic analysis for the C++ profiles framework
/// (P3589R2) and the built-in std::init initialization profile (P4222R1.1).
/// See clang/docs/ProfilesFramework.rst for the design.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SEMAPROFILES_H
#define LLVM_CLANG_SEMA_SEMAPROFILES_H

#include "clang/AST/ASTFwd.h"
#include "clang/Basic/Profiles.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/SemaBase.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace clang {

class AnalysisDeclContext;
class Module;
class ParsedAttr;
class ParsedAttributesView;
class ProfilesSuppressAttr;

class SemaProfiles : public SemaBase {
public:
  SemaProfiles(Sema &S);

  struct ProfileEnforcement : profiles::EnforcedProfile {
    SourceLocation EnforceLoc;
  };
  SmallVector<ProfileEnforcement, 4> EnforcedProfiles;

  /// True if an included AST file (PCH) contributed a non-empty top-level
  /// declaration to this TU. The [[profiles::enforce]] placement check
  /// (P3589R2 [decl.attr.enforce]p1) consults this instead of deserializing
  /// the PCH's declarations; ASTWriter ORs it forward so chained PCHs
  /// propagate the bit.
  bool TUPrecededByNonEmptyDecl = false;

  struct ProfileSuppressEntry {
    StringRef ProfileName;
    StringRef RuleName;
  };
  SmallVector<ProfileSuppressEntry, 4> ProfileSuppressStack;

  /// True while a class/constructor finalization profile callback runs.
  /// Finalization can fire as a side effect of instantiating an unrelated
  /// entity whose ProfileSuppressScope is still on ProfileSuppressStack, so
  /// during finalization that transient stack is ignored and suppression is
  /// resolved only from the finalized declaration and its lexical parents
  /// (token-based dominion, P3589R2 s2.4p3).
  bool InProfileFinalizationCheck = false;

  bool isProfileEnforced(StringRef ProfileName) const;

  /// True if any entry of \p Entries names an enforced profile. \p Entries is
  /// any profile opt-in table whose elements expose a \c Name member; shared
  /// by the post-parse dispatch gates (the CFG analysis pass guard and the
  /// finalization dispatcher).
  template <typename Table>
  bool anyProfileEnforced(const Table &Entries) const {
    return llvm::any_of(
        Entries, [&](const auto &E) { return isProfileEnforced(E.Name); });
  }

  const ProfileEnforcement *getProfileEnforcement(StringRef ProfileName) const;
  bool addProfileEnforcement(StringRef Name, StringRef Designator,
                             SourceLocation Loc);
  bool processProfilesEnforceAttr(
      const ParsedAttr &AL, Module *Mod, SmallVectorImpl<StringRef> *NewNames,
      SmallVectorImpl<StringRef> *NewDesignators,
      SmallVectorImpl<unsigned> *NewArgumentCounts = nullptr,
      SmallVectorImpl<StringRef> *NewArgumentKeys = nullptr,
      SmallVectorImpl<StringRef> *NewArgumentValues = nullptr,
      SmallVectorImpl<unsigned> *NewArgumentKinds = nullptr);

  ProfilesSuppressAttr *makeProfilesSuppressAttr(const ParsedAttr &AL);

  /// Create an implicit ProfilesSuppressAttr carrying just a profile and rule
  /// name (no justification or arguments), for propagating an active
  /// suppression onto a declaration.
  ProfilesSuppressAttr *makeImplicitProfilesSuppressAttr(StringRef ProfileName,
                                                         StringRef RuleName);

  bool isProfileSuppressed(StringRef ProfileName,
                           StringRef RuleName = "") const;
  bool isProfileSuppressed(StringRef ProfileName, StringRef RuleName,
                           const Decl *D) const;
  bool isProfileSuppressed(StringRef ProfileName, StringRef RuleName,
                           const Stmt *S, AnalysisDeclContext &AC) const;
  bool shouldEmitProfileViolation(StringRef ProfileName, StringRef RuleName,
                                  SourceLocation Loc);
  bool shouldEmitProfileViolation(StringRef ProfileName, StringRef RuleName,
                                  SourceLocation Loc, const Decl *D);
  bool shouldEmitProfileViolation(StringRef ProfileName, StringRef RuleName,
                                  const Stmt *UseStmt,
                                  AnalysisDeclContext &AC) const;
  bool checkProfileViolation(StringRef ProfileName, StringRef RuleName,
                             SourceLocation Loc, unsigned DiagID);

  /// Dispatch class-finalization profile callbacks for a completed class.
  /// Called from \c Sema::CheckCompletedCXXClass so parser, template
  /// instantiation, and lambda finalization paths all reach the same hook.
  /// Dependent, invalid, and lambda classes are filtered out.
  void checkProfileViolationsAtClassFinalization(CXXRecordDecl *RD);

  /// Dispatch constructor-finalization profile callbacks once a constructor's
  /// member-initializer list is complete. Called from \c ActOnMemInitializers
  /// and \c ActOnDefaultCtorInitializers, which also serve template
  /// instantiations (via \c InstantiateMemInitializers), so every
  /// user-defined constructor is covered at the point its \c inits() is fully
  /// populated -- unlike class finalization, which runs before any
  /// constructor body is parsed. Dependent, invalid, and delegating
  /// constructors are filtered out.
  void
  checkProfileViolationsAtConstructorFinalization(CXXConstructorDecl *Ctor);

  /// std::init / uninit_decl (R2, paper §4.2): diagnose an automatic variable
  /// definition that leaves the object (or a scalar subobject) indeterminate
  /// without an acknowledging [[uninit]] marker. Called from
  /// \c ActOnUninitializedDecl after default-initialization is attempted.
  void checkInitProfileUninitDecl(const VarDecl *Var);

  /// std::init / static_marker (paper §3, §4.2): diagnose [[uninit]] on a
  /// static or thread-storage variable, which is zero-initialized by language
  /// rule and therefore an initialized object. Called from
  /// \c ActOnUninitializedDecl.
  void checkInitProfileStaticMarker(const VarDecl *Var);

  /// std::init / static_runtime_init (paper §3): diagnose a non-local static
  /// whose initialization needs a runtime constructor. \p CheckConstInit
  /// lazily evaluates whether the initializer is constant (trivial
  /// default-init counts as constant here). Returns true if the diagnostic
  /// was emitted, in which case the caller skips -Wglobal-constructors.
  bool
  checkInitProfileStaticRuntimeInit(const VarDecl *Var,
                                    llvm::function_ref<bool()> CheckConstInit);

  /// std::init / uninit_with_initializer (R4): diagnose \p D if it is both
  /// marked [[uninit]] and has an initializer. Shared by the variable
  /// (\c CheckCompleteVariableDeclaration) and non-static data member
  /// (\c ActOnFinishCXXInClassMemberInitializer) paths. \p Init is the
  /// (possibly null) initializer; a RecoveryExpr placeholder for a failed
  /// initialization does not count as a user-written initializer.
  void checkInitProfileUninitWithInitializer(const ValueDecl *D,
                                             const Expr *Init);

  /// True if default-initialization of \p T would leave at least one scalar
  /// subobject with an indeterminate value. Shared by the std::init rules
  /// uninit_decl (at the variable declaration), ctor_uninit_member (for a
  /// class-typed member), and uninit_with_initializer. A class with a
  /// user-provided default constructor is trusted (that constructor is
  /// checked at its own definition). Dependent and incomplete types are
  /// treated as determinate.
  ///
  /// When \p HonorUninitMarkers is true, a data member marked [[uninit]]
  /// is treated as acknowledged and skipped, so a type whose only
  /// indeterminate scalars are all marked is reported as determinate.
  /// uninit_decl and ctor_uninit_member pass true (the marker excuses the
  /// member, paper §6.2); uninit_with_initializer passes false because it
  /// needs the factual answer (whether the default-initialization is
  /// genuinely a no-op).
  bool defaultInitLeavesScalarIndeterminate(QualType T,
                                            bool HonorUninitMarkers = false);

  /// If \p E (stripped of parens and implicit casts) directly names a
  /// declaration -- a DeclRefExpr or a MemberExpr -- return that declaration;
  /// otherwise null. The std::init checks read [[ref_to_uninit]] /
  /// [[uninit]] markers only off a directly named entity.
  static const ValueDecl *getDirectlyNamedDecl(const Expr *E);

  /// std::init / ref_to_uninit (paper §5): true only if \p E is affirmatively
  /// recognized as referring to (for a pointer source) or, when
  /// \p IsReference, denoting (for a glvalue source) uninitialized storage.
  /// Recognized purely locally from the expression's syntactic form -- the
  /// address of, or a subobject of, a [[uninit]] entity; a value of a
  /// [[ref_to_uninit]] pointer/reference or array; a dereference of such a
  /// pointer; a cast of such a pointer to another pointer type, or of such a
  /// glvalue to another reference; a call to a [[ref_to_uninit]]-returning
  /// function; or a new-expression whose default-initialization leaves the
  /// allocated object indeterminate (e.g. new int). A trusted-initialized
  /// source and an unrecognized (unknown) source both return false (no flow
  /// analysis).
  bool refersToUninitializedMemory(const Expr *E, bool IsReference) const;

  /// std::init / ref_to_uninit (paper §5): check that the initialization of a
  /// pointer or reference is consistent with its [[ref_to_uninit]] marking --
  /// a marked target must refer to uninitialized memory, and an unmarked
  /// target must not. Shared by the variable, data-member, assignment,
  /// argument, and return check sites; gated by shouldEmitProfileViolation.
  void checkInitProfileRefToUninit(SourceLocation Loc, bool TargetIsRefToUninit,
                            bool IsReference, const Expr *Src,
                            const Decl *D = nullptr);

  /// std::init / ref_to_uninit (paper §5): check that binding \p Src to
  /// \p Target (a variable, data member, parameter, or function) is
  /// consistent with the target's [[ref_to_uninit]] marking. \p T is the
  /// bound type -- the target's type, or the return type when \p Target is a
  /// function. No-op unless \p T is a non-dependent pointer or reference (a
  /// dependent type defers to instantiation, where the check site re-runs
  /// with the concrete type). \p D, when available, is the declaration used
  /// for suppression lookup and template-pattern deferral.
  void checkInitProfileRefToUninitBinding(SourceLocation Loc,
                                          const ValueDecl *Target, QualType T,
                                          const Expr *Src,
                                          const Decl *D = nullptr);

  /// std::init / uninit_read (paper §4.5): diagnose a read *through* a
  /// [[ref_to_uninit]] pointer or reference, whose result is itself
  /// uninitialized. Called from Sema::DefaultLvalueConversion at the single
  /// lvalue-to-rvalue chokepoint, with \p Glvalue the operand being loaded
  /// and \p ValueType its value type. Reuses the ref_to_uninit recognizer in
  /// read-only mode, so a direct read of a named [[uninit]] object is left to
  /// the flow-based uninit_read pass. A std::byte read is exempt (paper
  /// §4.5).
  void checkInitProfileReadThrough(SourceLocation Loc, const Expr *Glvalue,
                            QualType ValueType);

  /// std::init / pointer_marker + union_marker (paper §4.1, §5.6): diagnose
  /// [[uninit]] placed on a pointer, a union variable, or a union member.
  /// \p D must already carry the UninitAttr (the marker location is taken
  /// from it). Decl-aware via shouldEmitProfileViolation, so it defers on a
  /// templated pattern and is re-checked on the instantiated entity.
  void checkInitProfileMarkerPlacement(const Decl *D);

  /// [[ref_to_uninit]] is only meaningful on a pointer or reference to an
  /// object (for a function, its return type). Returns true when \p D's type
  /// is invalid for the marker, diagnosing err_ref_to_uninit_attr_invalid_type
  /// at \p AttrLoc unless \p Diagnose is false. A dependent type returns
  /// false: validation defers to the instantiation re-check in
  /// Sema::InstantiateAttrs, which drops the marker when the substituted type
  /// is invalid -- silently in a SFINAE context (\p Diagnose false there), so
  /// the marker can never affect overload resolution; a dropped marker is
  /// inert. Not profile policy -- fires regardless of -fprofiles, like the
  /// parse-time handler it serves.
  bool diagnoseInvalidRefToUninitMarker(const Decl *D, SourceLocation AttrLoc,
                                        bool Diagnose = true);

  class ProfileSuppressScope {
    Sema &S;
    unsigned Count = 0;

    void push(StringRef ProfileName, StringRef RuleName);
    void addFromDecl(const Decl *D);

  public:
    ProfileSuppressScope(Sema &S, const ParsedAttributesView &Attrs);
    ProfileSuppressScope(Sema &S, const Decl *D,
                         bool WalkLexicalParents = false);
    ProfileSuppressScope(Sema &S, ArrayRef<const Attr *> Attrs);
    ~ProfileSuppressScope();
  };
};

} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMAPROFILES_H

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
/// See clang/docs/ProfilesFrameworkInternals.rst for the design and
/// clang/docs/ProfilesFramework.rst for the user-facing documentation.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SEMAPROFILES_H
#define LLVM_CLANG_SEMA_SEMAPROFILES_H

#include "clang/AST/ASTFwd.h"
#include "clang/Basic/Profiles.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/SemaBase.h"
#include "llvm/ADT/DenseMap.h"
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
    /// Begin location of the construct the suppression appertains to (the
    /// declaration or statement, not the attribute). The entry's dominion
    /// starts here (P3589R2 s2.4p3).
    SourceLocation Begin;
    /// End location of the construct, recorded only when the construct was
    /// fully parsed at push time; invalid otherwise, leaving the dominion's
    /// end bounded by the ProfileSuppressScope's lifetime. That fallback is
    /// exact for a construct still being parsed -- its later tokens do not
    /// exist yet, and instantiation of a not-yet-defined template is
    /// deferred past the scope's death -- while a completed construct's
    /// recorded end keeps a live scope from covering a pattern first
    /// declared after it.
    SourceLocation End;
  };
  SmallVector<ProfileSuppressEntry, 4> ProfileSuppressStack;

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

  /// True if a live parse-time suppress entry for \p ProfileName /
  /// \p RuleName covers \p Loc. An entry matches only tokens within its
  /// construct's recorded range (its dominion, P3589R2 s2.4p3); when the
  /// construct was still being parsed at push time no end is recorded and
  /// the owning ProfileSuppressScope's lifetime bounds the dominion's end.
  /// Tokens from outside the construct -- e.g. a template pattern
  /// instantiated synchronously while the scope is live, wherever it is
  /// declared -- are not suppressed.
  bool isProfileSuppressed(StringRef ProfileName, StringRef RuleName,
                           SourceLocation Loc) const;
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

  /// P3589R2 [decl.attr.enforce]p5: a declaration and its redeclarations must
  /// appear in the dominions of mutually compatible profiles. Called from
  /// \c Sema::CheckRedeclarationInModule when \p New redeclares \p Old. Only
  /// a previous declaration from another module unit (a named module or a
  /// header unit) can carry a different dominion; that TU's dominion is
  /// approximated by the module's exported designator set. Profiles are
  /// compatible by name, with all std:: profiles mutually compatible.
  /// Diagnose-only: the redeclaration is not invalidated.
  void checkRedeclarationProfileCompatibility(const NamedDecl *New,
                                              const NamedDecl *Old);

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

  /// True if default-initialization of \p T is a genuine no-op that leaves
  /// the object (or every array element) uninitialized -- the only state
  /// consistent with an [[uninit]] marker. A non-trivial default constructor
  /// (user-provided anywhere in the subtree, a default member initializer, a
  /// virtual table pointer) initializes something, contradicting the marker
  /// (paper §4.2 rule 2, §5.3); a deleted or absent default constructor makes
  /// default-initialization ill-formed, not a no-op (the marker is
  /// unsatisfiable); an all-scalars-determinate type (e.g. an
  /// empty struct) has nothing uninitialized. Type-level (not a query on the
  /// synthesized construct-expression) because the field flavor and
  /// static_marker's no-initializer arm have no construct-expression, and
  /// getBaseElementType handles arrays and scalars uniformly. Shared by
  /// uninit_with_initializer and static_marker (through their common
  /// initializer guard) and the field-marker flavor.
  bool defaultInitIsVacuous(QualType T);

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
  /// function or to a known uninitialized-returning allocator (the malloc
  /// and alloca builtin families and raw replaceable ::operator new calls;
  /// calloc's result is initialized, realloc's
  /// unknown); or a new-expression whose default-initialization leaves the
  /// allocated object indeterminate (e.g. new int). A trusted-initialized
  /// source and an unrecognized (unknown) source both return false (no flow
  /// analysis).
  bool refersToUninitializedMemory(const Expr *E, bool IsReference) const;

  /// std::init / ref_to_uninit (paper §5): check that the initialization of a
  /// pointer or reference is consistent with its [[ref_to_uninit]] marking --
  /// a marked target must refer to uninitialized memory, and an unmarked
  /// target must not. Shared by the variable, data-member, assignment,
  /// argument, and return check sites; gated by shouldEmitProfileViolation.
  /// A Decl-less call defers only on an instantiation-dependent \p Src --
  /// such a construct is always rebuilt at instantiation, re-running this
  /// funnel with the substituted source -- and otherwise fires at definition
  /// time; if the construct is rebuilt at instantiation anyway (a local
  /// operand, a call argument, a return), the same diagnostic repeats there
  /// (accepted for now). A Decl-carrying call instead defers via the
  /// D->isTemplated() check in shouldEmitProfileViolation and fires on the
  /// instantiated declaration.
  void checkInitProfileRefToUninit(SourceLocation Loc, bool TargetIsRefToUninit,
                            bool IsReference, const Expr *Src,
                            const Decl *D = nullptr);

  /// std::init / ref_to_uninit (paper §5): check that binding \p Src to
  /// \p Target (a variable, data member, parameter, or function) is
  /// consistent with the target's [[ref_to_uninit]] marking. A null \p Target
  /// is a binding with no declaration to carry the marker (a parameter of a
  /// call through a function pointer) and is checked as unmarked. \p T is the
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
  /// and \p ValueType its value type; and from the compound-assignment
  /// (Sema::CheckAssignmentOperands) and increment/decrement
  /// (Sema::CreateBuiltinUnaryOp) operator sites, whose reads build no
  /// lvalue-to-rvalue node (the shift-compounds are excluded there because
  /// their LHS promotion already funnels through the chokepoint). Reuses the
  /// ref_to_uninit recognizer with its read access preset, so a direct read
  /// of a named [[uninit]] object is left to the flow-based uninit_read
  /// pass. A std::byte read is exempt (paper §4.5). Defers only on an
  /// instantiation-dependent \p Glvalue (rebuilt at instantiation, where the
  /// check re-runs); a non-dependent read fires at definition time and may
  /// repeat if the read is rebuilt at instantiation anyway (accepted).
  void checkInitProfileReadThrough(SourceLocation Loc, const Expr *Glvalue,
                            QualType ValueType);

  /// std::init / uninit_write (paper §5.4-§5.6): diagnose a scalar store to a
  /// proper subobject of a named [[uninit]] entity -- delayed piecemeal
  /// initialization, which only whole-object construct_at could make good.
  /// Called from Sema::CheckAssignmentOperands (the shared simple/compound
  /// assignment funnel) and from the built-in increment/decrement arm of
  /// Sema::CreateBuiltinUnaryOp, with \p LHS the store target. Reuses the
  /// recognizer with its write access preset: a store to the whole named
  /// entity is its initialization (paper §4.5), and storage reached through
  /// [[ref_to_uninit]] is trusted (the deferred construct_at slice), so only
  /// a below-top-level [[uninit]] marker fires. A std::byte store is exempt
  /// (paper §4.5). Defers only on an instantiation-dependent \p LHS (rebuilt
  /// at instantiation, where the check re-runs); a non-dependent store fires
  /// at definition time and may repeat if the assignment is rebuilt at
  /// instantiation anyway (accepted).
  void checkInitProfileSubobjectWrite(SourceLocation Loc, const Expr *LHS);

  /// std::init / ref_to_uninit (paper §5): a pointer argument passed through
  /// a variadic `...` parameter, which cannot carry [[ref_to_uninit]], is
  /// checked as an unmarked target. Called with the promoted argument from
  /// the C++ variadic promotion loops (Sema::GatherArgumentsForCall and
  /// Sema::BuildCallToObjectOfClassType); a non-pointer argument is a no-op
  /// (its value read is the lvalue-to-rvalue chokepoint's).
  void checkInitProfileVariadicArgument(const Expr *Arg);

  /// std::init / ref_to_uninit (paper §4.3): a by-reference lambda capture of
  /// \p Var binds a reference to its storage, and a capture cannot carry
  /// [[ref_to_uninit]], so capturing an entity that denotes uninitialized
  /// storage -- an [[uninit]] variable, or a [[ref_to_uninit]] reference --
  /// is always the unmarked-direction violation. Called from
  /// \c Sema::BuildLambdaExpr for each by-reference non-init variable capture
  /// (init-captures are checked at \c createLambdaInitCaptureVarDecl); defers
  /// only when the captured variable's type is instantiation-dependent.
  /// TreeTransform always rebuilds a lambda at instantiation, so a deferred
  /// capture re-processes there -- and a definition-time fire repeats there
  /// (accepted).
  void checkInitProfileRefCapture(SourceLocation Loc, const ValueDecl *Var);

  /// std::init / ref_to_uninit (paper §7.2): a member call binds its implicit
  /// object parameter to \p Object, and that parameter can never carry
  /// [[ref_to_uninit]], so a call on an object recognized as uninitialized
  /// storage is always the unmarked-direction violation. Called from
  /// \c Sema::PerformImplicitObjectArgumentInitialization, the funnel every
  /// member-call flavor's object argument converts through -- dot and arrow
  /// calls, member operators, functor operator(), operator->, and conversion
  /// operators. Explicit-object member functions initialize their object as an
  /// ordinary parameter and are already checked there; a destructor call is
  /// skipped (destruction of uninitialized storage is the deferred destroy_at
  /// slice), as is a static call operator (no implicit object parameter, like
  /// a static member call). Defers only on an instantiation-dependent
  /// \p Object -- the call
  /// is rebuilt at instantiation, re-running the funnel -- and otherwise fires
  /// at definition time, repeating if the call is rebuilt anyway (accepted).
  void checkInitProfileObjectArgument(const Expr *Object,
                                      const CXXMethodDecl *Method);

  /// std::init / ref_to_uninit (paper §4.3): assigning to a pointer must
  /// respect the assigned-to pointer's [[ref_to_uninit]] marking; a no-op for
  /// a non-pointer LHS. Hosts the cluster from Sema::CreateBuiltinBinOp's
  /// BO_Assign arm. An instantiation-dependent LHS defers to the
  /// instantiation rebuild (its marker cannot be read yet); the source's
  /// dependence is the shared funnel's to defer on.
  void checkInitProfilePointerAssignment(Expr *LHS, Expr *RHS,
                                         SourceLocation OpLoc);

  /// std::init: the check pair every built-in assignment hosts (paper
  /// §5.4-§5.6): the compound-assignment old-value load (read-through --
  /// excluding the shifts, whose LHS promotion already loads through the
  /// lvalue-to-rvalue chokepoint) and the subobject-write check. Hosts the
  /// cluster from Sema::CheckAssignmentOperands. \p IsCompound distinguishes
  /// `op=` from `=` (!CompoundType.isNull() at the host site).
  void checkInitProfileAssignmentOperands(BinaryOperatorKind Opc,
                                          Expr *LHSExpr, bool IsCompound,
                                          SourceLocation OpLoc);

  /// std::init: the check pair a built-in ++/-- hosts -- the old-value load
  /// (read-through) and the store (subobject-write). Hosts the cluster from
  /// Sema::CreateBuiltinUnaryOp's increment/decrement arm. Records the store
  /// credit last, after both pre-store checks.
  void checkInitProfileIncDec(Expr *Operand, SourceLocation OpLoc);

  /// std::init: record parse-order whole-entity store credit for \p LHS, the
  /// left operand of a completed built-in assignment (called from the tail
  /// of Sema::CheckAssignmentOperands) or the operand of a built-in ++/--.
  /// Assigning a whole [[uninit]] local is its initialization (paper
  /// §4.2/§4.5), and a store through the exact `*p` / `r` lvalue of a
  /// [[ref_to_uninit]] local or parameter initializes the pointee (§4.3); a
  /// store to a marked *pointer* itself reseats it and clears its pointee
  /// credit. Element stores (p[i] = e) neither credit nor invalidate
  /// (§5.4/§5.5 ban element-wise tracking), and escapes never credit (§6.2
  /// reserves callee-initialization for now_init()). Purely parse-order --
  /// no dominance or flow analysis -- so the credit errs only toward missed
  /// diagnostics. Deliberately not gated on enforcement or
  /// [[profiles::suppress]]: a suppressed store still initializes, and
  /// failing to credit it would turn suppression into later false positives.
  void recordInitProfileStore(const Expr *LHS);

  /// std::init / [[now_init]] (P4222R2 §6.2): a [[now_init]] callee
  /// initializes the storage bound to each of its [[ref_to_uninit]]
  /// parameters, so the binding earns the same parse-order credit the
  /// equivalent direct store would. Called from the tail of
  /// checkInitProfileRefToUninitBinding when \p Target is a marked parameter
  /// of a [[now_init]] function; recognizes the affirmatively creditable
  /// source shapes -- &u / u (whole-entity credit on an [[uninit]] local), p
  /// / *p / &*p (pointee credit on a marked local/parameter pointer; §6.2's
  /// initialize2(p) example), r (pointee credit on a marked reference), and
  /// &base.m / base.m (per-object member credit, resolveMemberStoreBase
  /// keys) -- through the recognizers' explicit-cast pass-through. Variadic
  /// arguments, unmarked parameters, and calls through function pointers
  /// never reach here (no marked ParmVarDecl target). Recorded regardless of
  /// enforcement, suppression, or diagnosis of the binding itself (the
  /// callee still initializes; recordInitProfileStore's rationale), but not
  /// in never-executed contexts.
  void recordNowInitArgument(const ValueDecl *Target, QualType T,
                             const Expr *Src);

  /// True if the current expression-evaluation context never executes at
  /// runtime (unevaluated or discarded-statement), mirroring
  /// shouldEmitProfileViolation's context checks: a store or a callee
  /// initialization seen there earns no credit. The shared gate of
  /// recordInitProfileStore and recordNowInitArgument.
  bool inNeverExecutedContext() const;

  /// True if \p VD is a local [[uninit]] variable credited by a recorded
  /// whole-entity store; the recognizers then classify it as initialized
  /// (which also enables the paper's reverse-direction rule: a credited
  /// entity requires an unmarked target).
  bool hasWholeObjectStoreCredit(const ValueDecl *VD) const;

  /// True if \p VD is a [[ref_to_uninit]] local/parameter pointer or
  /// reference credited by a recorded store through it; the storage behind
  /// it then classifies as initialized (until a pointer is reseated --
  /// references cannot be reseated, so their credit is never cleared).
  bool hasPointeeStoreCredit(const ValueDecl *VD) const;

  /// True if the [[uninit]] member \p F of the base object identified by
  /// \p Base (see resolveMemberStoreBase; null returns false) is credited by
  /// a recorded whole-member store; the member then classifies as
  /// initialized through that same base.
  bool hasMemberStoreCredit(const Decl *Base, const FieldDecl *F) const;

  /// Resolve the identity key of a member access's base object for the
  /// per-object member store credit: the parse-time pattern of the enclosing
  /// function declaration for a current-object access (this->m / m /
  /// (*this).m) -- so credit recorded in one function body can never satisfy
  /// a binding in another, while a statement an instantiation reuses
  /// (unrebuilt) from its template or generic-lambda pattern agrees with a
  /// rebuilt one on the key -- or the directly named local-storage,
  /// non-reference VarDecl of a dot access (a.m). Any other base -- another
  /// member (a.b.m; §5.4 rejects deep delayed-initialization tracking), an
  /// arrow through an arbitrary pointer value, a reference (an alias to an
  /// object also reachable other ways) -- is untrackable per object: null.
  const Decl *resolveMemberStoreBase(const MemberExpr *ME) const;

  /// Store-credit bits for recordInitProfileStore.
  enum InitStoreCreditFlags : unsigned {
    /// The [[uninit]] entity itself was assigned (u = e, u @= e, ++u).
    WholeStored = 1u << 0,
    /// The storage behind the [[ref_to_uninit]] entity was written through
    /// the exact *p / r lvalue.
    PointeeStored = 1u << 1,
  };

  /// Parse-order store credit, keyed by the credited local/parameter (only
  /// local-storage VarDecls carrying the relevant marker are ever inserted).
  /// Never cleared across the translation unit: the keys are unique
  /// declarations, and template instantiations build fresh declarations, so
  /// pattern-time and instantiation-time state stay independent.
  llvm::DenseMap<const VarDecl *, unsigned> InitStoreCredit;

  /// Parse-order whole-member store credit, keyed per base object: the base
  /// is the directly named local-storage VarDecl (a.m = e) or, for the
  /// current object (this->m = e / m = e), the parse-time pattern of the
  /// enclosing function declaration -- so credit recorded in one function
  /// body can never satisfy a binding in another, two locals of the same
  /// type never share credit, and instantiations agree with their pattern
  /// on statements they reuse from it (see resolveMemberStoreBase). Only
  /// WholeStored is ever set: member *pointee* stores (*a.p = e) are
  /// deliberately never credited -- per-object pointee aliasing (copies
  /// share pointees) makes them unsound to approximate.
  /// Never cleared, for the same reasons as InitStoreCredit (instantiations
  /// key on fresh field and function declarations).
  llvm::DenseMap<std::pair<const Decl *, const FieldDecl *>, unsigned>
      MemberStoreCredit;

  /// std::init / ref_to_uninit (paper §5): a thrown pointer copy-initializes
  /// the exception object, which cannot carry [[ref_to_uninit]]; a no-op for
  /// a non-pointer exception object. Hosts the cluster from
  /// Sema::BuildCXXThrow.
  void checkInitProfileThrowOperand(const Expr *Operand);

  /// std::init / ref_to_uninit (paper §5): a written initializer for an
  /// allocated pointer binds it like a variable initialization, and a heap
  /// pointer object cannot carry [[ref_to_uninit]]. \p Init is the single
  /// written initializer expression, or null when there is none (a no-op).
  /// Hosts the cluster from Sema::BuildCXXNew, which calls it for scalar
  /// allocations only: an array new's written elements are each checked by
  /// the aggregate element hooks instead. An instantiation-dependent
  /// allocated type defers to the instantiation rebuild.
  void checkInitProfileNewInitializer(QualType AllocType, Expr *Init);

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

  /// [[uninit]] is meaningless on a reference (it must bind when declared).
  /// Returns true when \p D's type is a reference, diagnosing
  /// err_uninit_attr_invalid_subject at \p AttrLoc unless \p Diagnose is
  /// false. A dependent type returns false: validation defers to the
  /// instantiation re-check in Sema::InstantiateAttrs, which drops the marker
  /// when the substituted type is a reference (silently in a SFINAE context).
  /// The parameter / structured-binding rejections stay in the parse-time
  /// handler -- they do not depend on the type. Not profile policy -- fires
  /// regardless of -fprofiles.
  bool diagnoseInvalidUninitMarker(const Decl *D, SourceLocation AttrLoc,
                                   bool Diagnose = true);

  class ProfileSuppressScope {
    Sema &S;
    unsigned Count = 0;

    void push(StringRef ProfileName, StringRef RuleName, SourceLocation Begin,
              SourceLocation End);
    void addFromDecl(const Decl *D);

  public:
    ProfileSuppressScope(Sema &S, const ParsedAttributesView &Attrs);
    ProfileSuppressScope(Sema &S, const Decl *D,
                         bool WalkLexicalParents = false);
    ProfileSuppressScope(Sema &S, ArrayRef<const Attr *> Attrs,
                         SourceLocation Begin, SourceLocation End);
    ~ProfileSuppressScope();
  };
};

} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMAPROFILES_H

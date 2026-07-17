====================================
C++ Profiles Framework Internals
====================================

.. contents::
   :depth: 2
   :local:


This document describes the implementation of the C++ Profiles framework
(`P3589R2 <https://open-std.org/JTC1/SC22/WG21/docs/papers/2025/p3589r2.pdf>`_)
for Clang contributors -- in particular, how to add a new profile.  For
user-facing documentation of the feature, see :doc:`ProfilesFramework`.


Architecture
============

The framework is profile-agnostic.  Profile names are opaque strings and
there is no central registry: a profile is "enforced" simply because the user
wrote ``[[profiles::enforce(name)]]``, and each rule within a profile is just
a string identifier that ``[[profiles::suppress(name, rule: "...")]]`` can
target.  ``SemaProfiles`` owns all the bookkeeping -- attribute parsing and
placement checking, enforcement tracking, suppression scoping, template
instantiation, module propagation, and PCH/BMI serialization -- so a profile
implementation consists only of its diagnostics plus calls to the framework
at its semantic check sites.

Profile-rule diagnostics are defined with the ``ProfileRuleError`` diagnostic
class rather than ``Error``.  It marks them SFINAE-suppressed: they do not
count as substitution failures and cannot change overload resolution, but
selected specializations replay them when actually used.  The framework
passes the profile name as ``%0``:

.. code-block:: text

   def err_profile_type_cast_reinterpret : ProfileRuleError<
     "'reinterpret_cast' is unsafe under profile '%0'">;

There are four implementation patterns, keyed on when the rule can be
checked.


Pattern 1: Parse-Time Check Sites
=================================

For a rule checkable at a single semantic entry point, the entire profile
implementation is one call at that site:

.. code-block:: c++

   checkProfileViolation("my::profile", "my_rule", Loc,
                         diag::err_my_profile_rule);

The call checks that the profile is enforced and not suppressed (via the
parse-time suppress stack) and skips unevaluated and discarded-statement
contexts.  ``test::type_cast`` is the in-tree example.

Inside a template, parse-time checks follow one unified model.  A
*non-dependent* construct is checked on the template pattern, at definition
time: instantiation may reuse such a node unchanged, so deferring would
silently lose the diagnostic.  This deliberately trades strict "as-if after
phase 7" purity (P3589R2 §1.1) for reuse-proof diagnostics.  An
*instantiation-dependent* construct is always rebuilt at instantiation, where
the re-run check sees the substituted form, once per specialization.  A
non-dependent construct that happens to be rebuilt anyway repeats its
definition-time diagnostic with an ``in instantiation of ...`` note -- an
accepted duplication.  Under ``-fdelayed-template-parsing`` the body of a
never-instantiated template is never parsed, so definition-time diagnosis of
non-dependent violations does not occur in that mode.


Pattern 2: Post-Parse / CFG-Based
=================================

For a rule that needs whole-function analysis.  Each post-parse analysis owns
a small opt-in table of the profiles that ride it, one row per profile
(profile name, rule name, diagnostic); the framework never learns the
profile's name.  ``test::uninit_read`` is the in-tree example:

.. code-block:: c++

   constexpr CFGUninitProfileEntry CFGUninitProfiles[] = {
       {"my::profile", /*Rule=*/"", diag::err_my_profile_rule},
   };

The analysis's pass guard ORs in ``anyProfileEnforced(Table)`` so the pass
runs for an enforced profile even when the corresponding warning is
silenced, and the analysis's diagnostic reporter walks the table calling
``shouldEmitProfileViolation(Name, Rule, Stmt, AnalysisDeclContext)`` per use
site, emitting the entry's diagnostic (and skipping the default warning) when
it returns true.

That overload is the post-parse counterpart of the parse-time suppress
stack: by the time the analysis runs the stack has unwound, so it walks the
AST upward from the use site -- enclosing ``AttributedStmt``\ s,
``DeclStmt``-declared variables, and the lexical ``Decl`` chain -- for a
matching ``[[profiles::suppress]]``.


Patterns 3 and 4: Class and Constructor Finalization
====================================================

For rules that run once per completed class definition (pattern 3,
``test::class_final``) or once per user-defined constructor with its complete
member-initializer list (pattern 4, ``test::ctor_final``).  Both share one
dispatcher and one per-pass table shape:

.. code-block:: c++

   constexpr FinalizationProfile<CXXRecordDecl> ClassFinalizationProfiles[] = {
       {"my::profile", &runMyProfileCallback},
   };

The class hook runs from the single function every class-completion path
funnels through (parsing, template instantiation, lambda completion); the
constructor hook runs from the two functions every constructor's
member-initializer list funnels through, including instantiation.  The
dispatchers filter out dependent entities (the hooks re-fire on each
instantiation), invalid ones, lambdas (pattern 3), and delegating
constructors (pattern 4).  Each callback gates its diagnostics on the
decl-aware ``shouldEmitProfileViolation`` overload, which walks the finalized
declaration and its lexical parents for a suppression.

The split between the two patterns matters: class finalization runs *before
any constructor body or member-initializer list has been parsed*, so a
pattern-3 callback must not inspect a constructor's ``inits()``.  Rules that
depend on what a constructor initializes belong on pattern 4; rules that need
flow analysis belong on pattern 2.


Suppression Dominion Mechanics
==============================

A ``[[profiles::suppress]]`` attribute's dominion is the token range of the
construct it appertains to (the user-level rule is stated in
:doc:`ProfilesFramework`).  The parse-time suppress stack -- pushed and
popped by ``ProfileSuppressScope`` RAII guards in the parser and the
template-instantiation machinery -- enforces this positionally: each entry
records its construct's token range, and a violation matches an entry only if
its location falls within that range.  This keeps a live suppress scope from
leaking into code its tokens do not cover, which would otherwise happen in
two ways: a template pattern instantiated synchronously while the scope is
live (instantiated code retains the pattern's source locations), and a class
or constructor finalized as a side effect of such an instantiation.
Conversely, a local class or lambda defined *inside* the suppressed construct
is covered, whichever path re-enters it.

The range's end is recorded only when the construct was already fully parsed
when the entry was pushed.  For a construct still being parsed no end exists
yet (a mid-parse end location would be misleadingly early), so the entry's
scope lifetime bounds the dominion instead -- exact mid-parse, because the
construct's later tokens do not exist yet, and instantiation of a template
that has no definition yet is deferred past the scope's death.

Suppression written on a template pattern or its lexical parents is
re-established around instantiation, so it applies to instantiated code.  The
reverse is not propagated: a scope live at the *point of instantiation*
covers the trigger's tokens, not the pattern's, and the positional match
above keeps it from suppressing checks inside a synchronously instantiated
body, NSDMI, default argument, or marker re-check.


Modules and Serialization
=========================

``[[profiles::enforce]]`` on a module interface declaration records the
enforced designators on ``Module::EnforcedProfileDesignators``, which is what
``[[profiles::require]]`` on an import validates against.  A header unit
records enforcement the same way from the empty-declaration form P3589R2
prescribes for headers.  A non-partition implementation unit inherits the
interface's enforcements through its implicit import of the primary
interface.  A partition implementation unit does not implicitly import the
interface, whose BMI is normally built later, so inheritance there is
best-effort: enforcements are inherited only when the interface's BMI is
already resident, and it is never force-loaded nor its absence diagnosed -- a
missed diagnostic, never a change to the meaning of a well-formed program.
``[[profiles::enforce]]`` on a *non-interface* module-declaration is recorded
only translation-unit-locally and is invisible to importers.

Serialization is automatic for every profile: enforcements are written to a
PCH as ``ENFORCED_PROFILES`` records and restored on load, and
``Module::EnforcedProfileDesignators`` is written to a BMI as
``SUBMODULE_ENFORCED_PROFILES`` records within each submodule block.


Redeclaration Compatibility
===========================

P3589R2 [decl.attr.enforce]p5 requires a declaration and its redeclarations
to appear in the dominions of mutually compatible profiles.
``checkRedeclarationProfileCompatibility`` runs from the module-level
redeclaration funnel and checks the rule symmetrically in both directions.
It is a framework rule: a plain error, not suppressible with
``[[profiles::suppress]]``, and diagnose-only (the redeclaration still
merges).  Two profiles are compatible if they have the same name (designator
arguments configure a profile without changing its identity) or if both are
standard ``std::``-prefixed profiles.

The previous declaration's dominion is approximated by its top-level module's
exported designators, which is exact for declarations in the module purview.
Two cases have an *unknown* dominion and are skipped rather than guessed at
(a missed diagnostic, never a wrong one): a declaration in an explicit global
module fragment (the exported set does not cover it), and a previous
declaration from the same module family (the exported set under-approximates
the interface TU's dominion, which the current unit inherits anyway).  A
textual or PCH previous declaration shares the current TU's dominion and is
not checked; implicit template instantiations are exempt.


Test Profiles
=============

The four built-in ``test::`` profiles exist only to exercise the framework in
the test suite.  They are gated on the ``-cc1``-only
``-fprofiles-test-profiles`` flag: under ``-fprofiles`` alone their
designators are still parsed, recorded, and exported across modules, but
``isProfileEnforced`` reports any ``test::``-prefixed profile as not
enforced, so no ``test::`` rule ever fires.  Because that gate keys on the
``test::`` prefix, a new test-only profile must also live under ``test::``.

- ``test::type_cast`` -- pattern 1; diagnoses ``reinterpret_cast<>`` (the
  keyword form only).
- ``test::uninit_read`` -- pattern 2; rides the existing CFG
  uninitialized-variables analysis.
- ``test::class_final`` -- pattern 3; fires on completion of every non-lambda
  class, on instantiations rather than dependent patterns.
- ``test::ctor_final`` -- pattern 4; fires once per user-defined,
  non-delegating constructor.

The names ``test::other``, ``test::bounds``, ``test::new_profile``, and
``test::not_enforced`` are deliberately *not* implemented and appear in
negative tests as "some other profile" stand-ins; adding a real profile under
any of them would invalidate those tests.


The std::init Implementation Map
================================

``std::init`` (documented in :doc:`ProfilesFramework`) uses all four
patterns.  Its rules map to mechanisms as follows:

.. list-table::
   :header-rows: 1
   :widths: 24 12 64

   * - Rule
     - Pattern
     - Primary entry points
   * - ``uninit_read``
     - 2 and 1
     - ``CFGUninitProfiles`` row for local variables;
       ``checkInitProfileCtorBody`` and ``checkInitProfileLocalMembers``
       (definite-assignment dataflow over ``[[uninit]]`` members; the
       ctor-body pass's ``CallExpr`` arm turns a ``[[now_init]]`` call into a
       ``Gen`` bit for the current-object storage bound to the callee's
       marked parameters, P4222R2 §6.2);
       ``checkInitProfileReadThrough`` at the lvalue-to-rvalue chokepoint,
       plus compound-assignment and increment/decrement hooks
   * - ``uninit_decl``
     - 1
     - ``checkInitProfileUninitDecl``
   * - ``uninit_with_initializer``
     - 1
     - ``checkInitProfileUninitWithInitializer``
   * - ``static_runtime_init``
     - 1
     - ``checkInitProfileStaticRuntimeInit``
   * - ``static_marker``
     - 1
     - ``checkInitProfileStaticMarker``
   * - ``union_marker``, ``pointer_marker``
     - attribute handler (enforcement-gated)
     - ``checkInitProfileMarkerPlacement``
   * - ``ctor_uninit_member``
     - 4
     - ``ConstructorFinalizationProfiles`` row
   * - ``ref_to_uninit``
     - 1
     - ``checkInitProfileRefToUninit`` behind per-site wrappers (variable
       and member initialization, call arguments, returns, throws,
       new-initializers, captures, object arguments)
   * - ``uninit_write``
     - 1
     - ``checkInitProfileSubobjectWrite``

Two helpers are shared across the rules.  ``refersToUninitializedMemory``
classifies an expression as referring to initialized, uninitialized, or
unknown storage purely from its syntactic form (parse-order store credit
refines it, recorded by ``recordInitProfileStore`` and by the
``recordNowInitArgument`` / ``recordNowUninitArgument`` pair, which share
one argument-shape walk to add or withdraw the credit of storage a
``[[now_init]]`` callee initializes or a ``[[now_uninit]]`` callee
destroys); its ``UninitAccessOpts``
presets distinguish a *binding* source (markers count everywhere), a value
*read*, and a scalar *store* (which differ in whether the top-level
``[[uninit]]`` marker counts and whether ``[[ref_to_uninit]]`` storage is
trusted).  ``defaultInitLeavesScalarIndeterminate`` answers whether a type's
default-initialization leaves an unacknowledged scalar subobject
indeterminate, trusting user-provided default constructors -- the paper's
trust-the-constructor principle (P4222R1.1 §5.1), which is also why members
of objects initialized by a user-provided constructor are deliberately not
flow-tracked.

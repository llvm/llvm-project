===========================
C++ Profiles Framework
===========================

.. contents::
   :depth: 3
   :local:


Introduction
============

The C++ Profiles framework (`P3589R2
<https://open-std.org/JTC1/SC22/WG21/docs/papers/2025/p3589r2.pdf>`_) allows a
translation unit to opt into additional language restrictions called *profiles*.
A profile is a named set of rules enforced by the compiler. A translation unit
requests enforcement with ``[[profiles::enforce(...)]]``; individual
declarations or statements can suppress enforcement with
``[[profiles::suppress(...)]]``; and module imports can require that an imported
module enforces a profile with ``[[profiles::require(...)]]``.

Profiles do not change the meaning of well-formed programs with no undefined
behavior.  Their static semantic effects are conceptually applied only after
translation phase 7: a profile cannot change the outcome of overload resolution
or template instantiation, and it is not possible to SFINAE on a profile
violation.

The framework is profile-agnostic.  It handles attribute parsing, enforcement
tracking, suppression scoping, module integration, and serialization.
Individual profiles only need to call a single API at the appropriate semantic
check sites (``Sema::checkProfileViolation`` for parse-time checks, or
``Sema::shouldEmitProfileViolation`` from a per-pass dispatch table for
post-parse analyses).  Everything else -- suppression, template instantiation,
SFINAE exclusion, module propagation, and PCH/BMI serialization -- is handled
by the framework automatically.


Driver Flag
-----------

The entire framework is gated on the ``-fprofiles`` command-line flag, which
sets ``LangOpts.Profiles``.  The flag is C++-only (declared with
``ShouldParseIf<cplusplus.KeyPath>`` in ``clang/include/clang/Options/Options.td``)
and defaults to off.

.. code-block:: bash

   clang++ -std=c++23 -fprofiles example.cpp

Without ``-fprofiles``:

- ``[[profiles::enforce]]``, ``[[profiles::suppress]]``, and
  ``[[profiles::require]]`` are diagnosed as ``warn_attribute_ignored`` and
  have no semantic effect.
- No profile rule check ever fires, even at sites that call
  ``checkProfileViolation``.

The framework's parse-time bookkeeping (``ProfileSuppressScope``, attribute
custom parsing, etc.) is also no-ops when ``LangOpts.Profiles`` is false, so
the flag is the single switch that turns the entire feature on or off.


Attribute Reference
===================

The three attributes are spelled in the ``profiles`` scope and accept either
the ``[[profiles::name(...)]]`` or ``[[using profiles: name(...)]]`` syntax.
Each attribute requires an argument clause; ``[[profiles::enforce]]`` and
``[[profiles::require]]`` with no parentheses are diagnosed.

``[[profiles::enforce(profile-designator-list)]]``
   Allowed only on an *empty-declaration* at translation-unit scope or on a
   *module-declaration*.  At TU scope, it must precede every non-empty
   declaration in the translation unit.  Each profile-designator is a
   ``::``-separated identifier sequence optionally followed by an
   argument-clause (e.g. ``vendor(fortify: 3)``).  Repeating the same name
   with the same canonical designator is allowed; repeating it with a
   different canonical designator is an error.

``[[profiles::suppress(profile-name [, justification: "..."] [, rule: "..."])]]``
   Allowed on declarations and statements.  Suppresses violations of the named
   profile (optionally narrowed to a single rule) within the appertaining
   declaration or statement; see :ref:`profiles-token-dominion` below.  The
   ``justification:`` argument, if present, must be a string literal; the
   ``rule:`` argument may be a string literal or a bare token.  Both
   arguments are recorded but otherwise opaque to the framework.

``[[profiles::require(profile-designator)]]``
   Allowed only on a *module-import-declaration*.  Diagnoses if the imported
   module's exported enforced-profile set does not contain a designator
   matching the requested one.  Importing a module does **not** retroactively
   enforce its profiles in the importer.

See the auto-generated :doc:`AttributeReference` for the AttrDocs entries
linked from these attributes.


Implementing a New Profile
==========================

Adding a new profile requires no changes to the framework itself.  A profile is
defined entirely by:

1. A **profile name** (a ``::``-separated identifier sequence such as
   ``vendor::safety`` or ``std::type``).
2. Zero or more **rule names** (string identifiers such as
   ``"reinterpret_cast"``).
3. **Diagnostics** emitted when a rule is violated.
4. **Check sites** in the compiler where the framework is consulted.

There is no central registry of profiles.  The framework treats profile names
as opaque strings; a profile is considered "enforced" simply because the user
wrote ``[[profiles::enforce(name)]]`` in their source.  Each rule within a
profile is likewise just a string identifier; users can suppress individual
rules with ``[[profiles::suppress(profile_name, rule: "rule_name")]]``.

There are two implementation patterns, depending on when the rule is checked.


Define Diagnostics
------------------

Add diagnostics to ``clang/include/clang/Basic/DiagnosticSemaKinds.td`` in the
``// C++ Profiles framework (P3589R2)`` group.  Each diagnostic should accept
``%0`` for the profile name, since the framework passes the profile name as
the first diagnostic argument.  The group declares the helper class

.. code-block:: text

   class ProfileRuleError<string str> : Error<str> {
     let SFINAE = SFINAE_Suppress;
   }

so that profile-rule diagnostics participate in the SFINAE machinery as
suppressed errors -- they do not count as substitution failures and cannot
change overload resolution, but selected specializations replay them when
they are actually used.  Define new rules using ``ProfileRuleError`` rather
than ``Error`` directly:

.. code-block:: text

   def err_profile_type_cast_reinterpret : ProfileRuleError<
     "'reinterpret_cast' is unsafe under profile '%0'">;


Pattern 1: Sema Check-Site Profile
----------------------------------

Used when the rule can be checked at a single, well-defined parse-time site
in Sema (typically inside a ``Sema::Build*`` or ``Sema::Act*`` routine).
``test::type_cast`` is the in-tree example.

At each such site, call ``Sema::checkProfileViolation``:

.. code-block:: c++

   checkProfileViolation("my::profile", "my_rule", Loc,
                         diag::err_my_profile_rule);

This function checks whether the profile is enforced and not suppressed.
During template argument deduction, profile rule diagnostics are suppressed
by Clang's normal SFINAE machinery so they cannot affect overload resolution
or template instantiation; selected specializations replay suppressed
diagnostics when used.  Unevaluated and discarded-statement contexts are
skipped.  The profile name is passed as ``%0``.

Suppression for parse-time check sites is consulted via the
``ProfileSuppressStack`` maintained by the parser-side ``ProfileSuppressScope``
RAII guards (see :ref:`profiles-internals` below).  Profile implementers do
not need to interact with the stack directly.


Pattern 2: Post-Parse / CFG-Based Profile
-----------------------------------------

Used when the rule cannot be checked at a single Sema entry point because it
depends on whole-function analysis -- typically a CFG-based analysis run
after a function body is complete.  ``test::uninit_read`` is the in-tree
example: it diagnoses reads of uninitialized variables on top of Clang's
existing CFG-based uninitialized-variables analysis.

This pattern needs three pieces, all colocated with the analysis pass (the
framework intentionally does not learn the profile name).

1. **Add the profile to the analysis pass's per-pass opt-in table.**
   Each post-parse analysis owns a small table of the profiles that ride it,
   one row per profile (profile name, rule name, diagnostic id).  The
   in-tree example is ``CFGUninitProfiles`` in
   ``clang/lib/Sema/AnalysisBasedWarnings.cpp``:

   .. code-block:: c++

      struct CFGUninitProfileEntry {
        StringRef Name;
        StringRef Rule;
        unsigned DiagID;
      };
      constexpr CFGUninitProfileEntry CFGUninitProfiles[] = {
          {"my::profile", /*Rule=*/"", diag::err_my_profile_rule},
      };

2. **Gate the analysis pass on the table.**
   Analysis-based passes in ``AnalysisBasedWarnings.cpp::IssueWarnings`` are
   normally run only when their corresponding warning flag is enabled.  To
   run the pass for an enforced profile even when the underlying warning is
   silenced, OR a ``llvm::any_of(Table, [&](const auto &E) { return
   S.isProfileEnforced(E.Name); })`` check into the existing pass guard.
   The in-tree example wraps this as ``anyCFGUninitProfileEnforced(S)`` and
   the pass guard becomes
   ``anyCFGUninitProfileEnforced(S) || !Diags.isIgnored(...)``.

3. **Walk the table in the analysis's diagnostic reporter.**
   For each use site the analysis would have warned about, iterate the
   table and call
   ``Sema::shouldEmitProfileViolation(name, rule, Stmt*, AnalysisDeclContext&)``,
   which walks parent statements and lexical declaration contexts to honor
   ``[[profiles::suppress]]`` on enclosing AST nodes (the post-parse
   counterpart to ``ProfileSuppressStack``).  Emit the entry's diagnostic
   when it returns true, and skip the default warning path.  In the
   in-tree example (``UninitValsDiagReporter`` in
   ``AnalysisBasedWarnings.cpp``):

   .. code-block:: c++

      for (const auto &U : *vec) {
        for (const CFGUninitProfileEntry &E : CFGUninitProfiles) {
          if (!S.shouldEmitProfileViolation(E.Name, E.Rule, U.getUser(), AC))
            continue;
          S.Diag(U.getUser()->getBeginLoc(), E.DiagID)
              << E.Name << vd->getDeclName();
          S.Diag(vd->getLocation(), diag::note_var_declared_here)
              << vd->getDeclName();
          return;
        }
      }

The diagnostic itself is defined with ``ProfileRuleError`` as in pattern 1.

The post-parse Stmt-walking suppression check is intentionally separate from
the parse-time stack check: by the time CFG analysis runs, the parse stack
has been unwound, so the framework instead walks the AST.  The Stmt-walking
overload of ``isProfileSuppressed`` examines:

- ``AttributedStmt`` ancestors of the use site.
- ``DeclStmt`` ancestors (whose declared ``VarDecl``\ s carry the attribute,
  not the enclosing statement).
- The enclosing ``Decl`` chain via ``getLexicalDeclContext()``.


.. _profiles-token-dominion:

Suppression Dominion is Token-Based
===================================

A ``[[profiles::suppress(P)]]`` attribute suppresses profile ``P`` in the
token range of the declaration or statement it appertains to -- nothing more.
For a variable declaration that range covers the initializer expression (so
``[[profiles::suppress(P)]] T x = init();`` silences violations inside
``init()``), but it does *not* tag the variable as permitted-uninitialized for
subsequent uses; those uses appear in different declarations or statements
and are checked normally at their own source location. Profiles that need
per-object "opt-out of this check everywhere this value is used" semantics
(for example, the proposed ``[[uninitialized]]`` attribute of the
initialization profile) must introduce their own, separate, decl-scoped
marker.

This applies identically to the parse-time suppression stack and the
post-parse Stmt-tree walker described in pattern 2.


.. _profiles-internals:

Framework Internals Reference
=============================

This section describes the framework mechanisms that profile implementers
benefit from understanding, even though they do not interact with them directly.

``ProfileSuppressScope``
------------------------

An RAII guard that pushes suppression entries onto ``Sema::ProfileSuppressStack``
and pops them on destruction.  It is used by the parser and template
instantiation machinery to make ``[[profiles::suppress]]`` attributes active
during the appropriate region.  ``checkProfileViolation`` consults
``ProfileSuppressStack`` directly, so profile implementers never need to create
``ProfileSuppressScope`` objects.

Stmt-Tree Suppression Walker
----------------------------

The ``isProfileSuppressed(name, rule, Stmt*, AnalysisDeclContext&)`` overload
is the post-parse counterpart to ``ProfileSuppressStack``.  It is used by
analyses that run after parsing (when the parse-time stack no longer
reflects the enclosing region) and walks the AST upward from a use site to
find any matching ``[[profiles::suppress]]`` attribute on an enclosing
``AttributedStmt``, ``DeclStmt``-declared ``VarDecl``, or lexical
``Decl`` parent.

Template Instantiation
----------------------

During template instantiation, the framework ensures that
``[[profiles::suppress]]`` on the template pattern and its lexical parents
applies to instantiated code.  This is done via ``ProfileSuppressScope`` with
``WalkLexicalParents=true`` at several sites:

- ``SemaTemplateInstantiateDecl.cpp`` -- function and variable template
  instantiation.
- ``SemaTemplateInstantiate.cpp`` -- default member initializer instantiation.
- ``TreeTransform.h`` -- ``TransformAttributedStmt`` (suppress on statements)
  and ``TransformDeclStmt`` (suppress on declarations within a ``DeclStmt``).

Module Enforcement
------------------

``[[profiles::enforce(...)]]`` on a module interface declaration records the
enforced profile designators on ``Module::EnforcedProfileDesignators``.  Module
implementation units automatically inherit the interface's enforcements.
``[[profiles::require(...)]]`` on an import-declaration validates that the
imported module's ``EnforcedProfileDesignators`` contains a matching designator.

Importing a module that enforces a profile does **not** enforce that profile in
the importing translation unit.  Enforcement is always explicit and local.

Serialization
-------------

The framework serializes enforcement state automatically.  Profile implementers
do not need to add any serialization code.

- **PCH**: ``Sema::EnforcedProfiles`` is written as ``ENFORCED_PROFILES``
  records in the AST bitstream and restored when the PCH is loaded.
- **Module BMI**: ``Module::EnforcedProfileDesignators`` is written as
  ``SUBMODULE_ENFORCED_PROFILES`` records within each submodule block.

Intentional Omissions
=====================

The following parts of P3589R2 are deliberately not implemented:

- ``[[profiles::exempt(...)]]`` (P3589R2 section 1.1.6), which would exempt
  named included source files from profile enforcement. Implementing it
  requires bookkeeping that connects the original spelling of an ``#include``
  to the source locations of constructs in the included file, and the feature
  is not needed to exercise or validate the rest of the framework.
- The redeclaration consistency rule from P3589R2 section 2.2 paragraph 5
  (every redeclaration of a declaration in the dominion of a profile must
  itself appear in the dominion of a compatible profile). Profile attributes
  on redeclarations are parsed and recorded, but no cross-redeclaration
  compatibility check is performed.


Built-in Test Profiles
======================

The tree ships two minimal, test-only profiles -- one per implementation
pattern -- so the framework's behavior can be exercised without depending on
any user-facing profile.  Both are gated on ``-fprofiles``.

By convention:

- Real test profiles live under the ``test::`` namespace.  Today there are
  two: ``test::type_cast`` and ``test::uninit_read``.
- The names ``test::other``, ``test::bounds``, ``test::new_profile``, and
  ``test::not_enforced`` are deliberately *not* implemented and appear only
  in negative tests as stand-in "some other profile" names.  Adding a real
  profile under any of these names would invalidate those tests.


The ``test::type_cast`` Profile
-------------------------------

A pattern-1 (Sema check-site) profile.  Demonstrates the simple case where
a rule can be checked from a single Sema entry point.

- **Rules**: ``reinterpret_cast``.
- **Diagnostic**: ``err_profile_type_cast_reinterpret``
  ("'reinterpret_cast' is unsafe under profile '%0'").
- **Check site**: ``Sema::BuildCXXNamedCast`` in ``clang/lib/Sema/SemaCast.cpp``,
  inside the ``reinterpret_cast`` arm.

The entire profile implementation is the single call:

.. code-block:: c++

   checkProfileViolation("test::type_cast", "reinterpret_cast", OpLoc,
                         diag::err_profile_type_cast_reinterpret);


The ``test::uninit_read`` Profile
---------------------------------

A pattern-2 (post-parse / CFG-based) profile.  Demonstrates the case where
the rule depends on whole-function analysis and must run after parsing.
It diagnoses reads of uninitialized variables by reusing Clang's existing
CFG-based uninitialized-variables analysis.

- **Rules**: none (the profile has a single implicit rule, so the rule
  string is empty).
- **Diagnostic**: ``err_profile_uninit_read``
  ("variable %1 is read before initialization under profile '%0'").
  A companion ``note_var_declared_here`` is emitted at the variable's
  declaration.
- **Opt-in table**: ``CFGUninitProfiles`` in
  ``clang/lib/Sema/AnalysisBasedWarnings.cpp``.  The ``IssueWarnings`` pass
  guard consults it via ``anyCFGUninitProfileEnforced(S)`` so the analysis
  runs even when ``-Wuninitialized`` is silenced, and
  ``UninitValsDiagReporter::diagnoseUnitializedVar`` walks it *before* the
  default warning path -- when an entry's
  ``Sema::shouldEmitProfileViolation`` returns true the entry's diagnostic
  fires and the default warning is skipped entirely.

The Stmt-tree suppression walker is what makes ``[[profiles::suppress]]``
work for this profile: by the time the CFG analysis runs, the parse-time
``ProfileSuppressStack`` has been unwound, so the helper consults the AST
directly via ``shouldEmitProfileViolation(name, rule, Stmt*, AnalysisDeclContext&)``.


In-Tree Tests
=============

These tests collectively exercise the framework and the two built-in
profiles.  When changing the framework, run them all with
``check-clang-sema``, ``check-clang-parser``, and ``check-clang-pch``.

- ``clang/test/Parser/cxx-profiles-framework.cpp`` -- attribute parser:
  valid ``enforce``/``suppress``/``require`` forms, the ``[[using profiles:
  ...]]`` syntax, profile-name and profile-argument grammar, and the
  parse-error / missing-argument-clause paths.
- ``clang/test/SemaCXX/safety-profile-framework.cpp`` -- attribute placement
  and basic semantic checks (``enforce`` only on empty-declarations at TU
  scope, ``require`` only on imports, ``suppress`` on declarations and
  statements, ``justification:`` must be a string literal, etc.).
- ``clang/test/SemaCXX/safety-profile-framework-modules.cppm`` -- module
  integration: ``enforce`` on a module-declaration is exported via the BMI,
  ``require`` validates against an imported module's exported set, GMF-only
  ``enforce`` does not leak through the BMI, interface-to-implementation
  propagation, partition interfaces, and the without-``-fprofiles`` ignored
  paths.
- ``clang/test/SemaCXX/safety-profile-type-cast.cpp`` -- the
  ``test::type_cast`` profile: enforcement, suppression on every supported
  declaration and statement form, template instantiation, SFINAE
  exclusion, lambdas (including generic lambdas with suppression carried
  through instantiation), and out-of-line members of suppressed classes
  and namespaces.
- ``clang/test/SemaCXX/safety-profile-uninit-read.cpp`` -- the
  ``test::uninit_read`` profile.  Cases are gated on ``-DCASE=N`` so the
  analysis-based-warnings early-exit-on-first-error does not hide later
  cases; case 0 is the no-violation baseline used by both the
  ``-fprofiles`` and the without-``-fprofiles`` runs.
- ``clang/test/PCH/cxx-profiles-enforce.cpp`` -- ``[[profiles::enforce]]``
  state survives PCH serialization round-trip.

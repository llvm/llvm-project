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
check sites (``SemaProfiles::checkProfileViolation`` for parse-time checks, or
``SemaProfiles::shouldEmitProfileViolation`` from a per-pass dispatch table for
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
- Their argument clauses are **not** checked against the P3589R2 profile
  grammar: like any standard attribute the implementation does not act on,
  an arbitrary balanced-token argument clause -- or none at all -- is
  accepted, so code annotated for a profiles-enabled build compiles cleanly
  (modulo the warning) with the feature off.  P3589R2's grammar is enforced
  only under ``-fprofiles``.
- No profile rule check ever fires, even at sites that call
  ``checkProfileViolation``.

The framework's parse-time bookkeeping (``ProfileSuppressScope``, attribute
custom parsing, etc.) is also no-ops when ``LangOpts.Profiles`` is false, so
the flag is the single switch that turns the entire feature on or off.

The built-in ``test::`` profiles (see `Built-in Profiles`_) exist only to
exercise the framework and are additionally gated on the ``-fprofiles-test-profiles``
flag, which sets ``LangOpts.ProfilesTestProfiles``.  This flag is ``-cc1``-only
(not exposed by the driver) and is intended solely for running the test suite.
Under ``-fprofiles`` alone, ``[[profiles::enforce(test::...)]]`` is still
recognized (it is not ``warn_attribute_ignored``) and its designator is still
recorded and exported across modules, but ``SemaProfiles::isProfileEnforced`` reports
any ``test::``-prefixed profile as not enforced, so no ``test::`` rule ever
fires.  Real profiles such as ``std::init`` are unaffected by this flag.


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

At each such site, call ``SemaProfiles::checkProfileViolation``:

.. code-block:: c++

   checkProfileViolation("my::profile", "my_rule", Loc,
                         diag::err_my_profile_rule);

This function checks whether the profile is enforced and not suppressed.
During template argument deduction, profile rule diagnostics are suppressed
by Clang's normal SFINAE machinery so they cannot affect overload resolution
or template instantiation; selected specializations replay suppressed
diagnostics when used.  Unevaluated and discarded-statement contexts are
skipped.  The profile name is passed as ``%0``.

``checkProfileViolation`` fires at parse time.  Inside a template, parse-time
checks follow one unified model.  A *non-dependent* construct is checked on
the template *pattern*, at definition time: TreeTransform may return such a
node unchanged at instantiation (for some node kinds, such as casts, a
non-dependent ``Build*`` result is reused), so deferring would silently lose
the diagnostic.  This deliberately trades strict "as-if after phase 7" purity
(P3589R2 §1.1) for reuse-proof diagnostics: a non-dependent violation
diagnoses even in a never-instantiated template or in an ``if constexpr``
branch whose discarding is not yet known at the pattern (a branch already
known discarded -- a non-value-dependent false condition -- stays silent).
An *instantiation-dependent* construct cannot be checked on the pattern; it
is always rebuilt at instantiation, where the re-run ``Build*`` checks the
substituted form, once per specialization.  A construct with non-dependent
check operands can still be rebuilt at instantiation (a local variable, a
call argument, or a return statement forces a rebuild, for example); the
re-run ``Build*`` then repeats the definition-time diagnostic at the same
location, with an ``in instantiation of ...`` note.  This repetition is
accepted for now; ``test::type_cast``'s cast nodes are never rebuilt when
non-dependent, so it never repeats.

Under ``-fdelayed-template-parsing`` the body of a never-instantiated
template is never parsed at all, so definition-time diagnosis of
non-dependent violations does not occur in that mode.

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
   silenced, OR an ``S.anyProfileEnforced(Table)`` check (the shared
   ``SemaProfiles::anyProfileEnforced`` gate, also used by the finalization dispatch)
   into the existing pass guard.  The in-tree example's pass guard becomes
   ``hasEnforcedCFGUninitProfile() || !Diags.isIgnored(...)`` (a small
   accessor over ``S.anyProfileEnforced(CFGUninitProfiles)``).

3. **Walk the table in the analysis's diagnostic reporter.**
   For each use site the analysis would have warned about, iterate the
   table and call
   ``SemaProfiles::shouldEmitProfileViolation(name, rule, Stmt*, AnalysisDeclContext&)``,
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


Pattern 3: Class-Finalization Profile
-------------------------------------

Used when the rule applies to a class as a whole and needs to run once after
the class definition is complete -- for example, "every non-private field of
this class must satisfy property X" or "this class's field set must look
like Y."  ``test::class_final`` is the in-tree example.

The dispatch point is ``SemaProfiles::checkProfileViolationsAtClassFinalization``,
called from the end of ``Sema::CheckCompletedCXXClass`` in
``clang/lib/Sema/SemaDeclCXX.cpp``.  ``CheckCompletedCXXClass`` is the
single function reached from every class-completion path -- the parser
(``ActOnFinishCXXMemberSpecification``), template instantiation
(``InstantiateClass``), and lambda completion -- so wiring the hook there
covers all of them with no extra plumbing.

The class-finalization entry point
``checkProfileViolationsAtClassFinalization`` filters out classes the rules
are not meant to see:

- Dependent classes (``isDependentType()``).  The hook will re-fire on each
  instantiation via the template-instantiation completion path.
- Invalid classes (``isInvalidDecl()``).
- Lambdas (``isLambda()``).  Closure types have no user-controlled field
  shape, so class-finalization rules do not apply.

This pattern needs two pieces, both colocated with the dispatcher.

1. **Add the profile to the class-finalization opt-in table.**
   ``ClassFinalizationProfiles`` in ``clang/lib/Sema/SemaProfiles.cpp`` is a
   small per-pass table of profile name plus callback.  One row per profile:

   .. code-block:: c++

      // FinalizationProfile<Node> is shared with pattern 4.
      template <class Node> struct FinalizationProfile {
        StringRef Name;
        void (*Callback)(Sema &, Node *);
      };
      constexpr FinalizationProfile<CXXRecordDecl> ClassFinalizationProfiles[] = {
          {"my::profile", &runMyProfileCallback},
      };

   The shared ``dispatchFinalizationProfiles`` dispatcher (used by both
   patterns 3 and 4) checks ``anyProfileEnforced(Table)``, iterates the
   table, skips entries whose profile is not enforced, and invokes the
   callback.  Each callback passes the finalized ``Decl`` (here the
   ``CXXRecordDecl``) to the decl-aware ``shouldEmitProfileViolation``
   overload, which walks the declaration and its lexical parents for a
   matching ``[[profiles::suppress]]``, so suppression on the class or any
   enclosing lexical ``Decl`` works without the dispatcher establishing a
   suppress scope.  Finalization can run as a side effect of an *unrelated*
   template instantiation whose ``[[profiles::suppress]]`` scope is still on
   the transient parse-time ``ProfileSuppressStack``; because stack entries
   are matched against the violation's location (see
   :ref:`profiles-token-dominion`), such a scope -- whose construct's tokens
   do not cover the finalized class -- does not suppress the callback's
   diagnostics.

2. **Emit diagnostics from the callback via**
   ``SemaProfiles::shouldEmitProfileViolation``.  Each callback decides where on
   the class to point and which diagnostic to use, possibly with notes:

   .. code-block:: c++

      void runMyProfileCallback(Sema &S, CXXRecordDecl *RD) {
        if (!S.shouldEmitProfileViolation("my::profile", /*Rule=*/"",
                                          RD->getLocation()))
          return;
        S.Diag(RD->getLocation(), diag::err_my_profile_rule)
            << "my::profile" << RD;
      }

The diagnostic itself is defined with ``ProfileRuleError`` as in patterns 1
and 2.

Class-finalization is for **structural** rules -- those answerable from the
class's declared members, their types, and their attributes.  The callbacks
run while the ``CXXRecordDecl`` is being finalized (immediately before
``CheckCompletedCXXClass`` returns), which is *before any constructor body or
member-initializer list has been parsed* -- inline member bodies are
late-parsed afterward, and out-of-line and template member constructors later
still.  A class-finalization callback therefore must not inspect a
constructor's ``inits()`` (they are empty here).  Rules that depend on what a
constructor initializes belong on the constructor-finalization dispatch
(pattern 4); rules that need whole-function flow analysis belong on a
post-parse CFG pass (pattern 2).


Pattern 4: Constructor-Finalization Profile
-------------------------------------------

Used when the rule applies to a single constructor and needs that
constructor's complete member-initializer list -- for example, "every member
must be initialized by this constructor."  ``test::ctor_final`` is the in-tree
example, and the ``std::init`` ``ctor_uninit_member`` rule is the real one.

The dispatch point is ``SemaProfiles::checkProfileViolationsAtConstructorFinalization``,
called right after ``DiagnoseUninitializedFields`` in
``Sema::ActOnMemInitializers`` and ``Sema::ActOnDefaultCtorInitializers`` in
``clang/lib/Sema/SemaDeclCXX.cpp``.  Those two functions are the funnel for
every user-defined constructor -- written or implicit member-initializer
list, inline or out-of-line -- and template instantiation reaches the first
of them through ``Sema::InstantiateMemInitializers``, so the hook sees every
constructor at the point its ``inits()`` (including synthesized entries) is
complete.

The constructor-finalization entry point
``checkProfileViolationsAtConstructorFinalization`` filters out constructors
the rules are not meant to see:

- Dependent constructors (``isDependentContext()``).  The hook re-fires on
  each instantiation.
- Invalid constructors (``isInvalidDecl()``).
- Delegating constructors (``isDelegatingConstructor()``), which leave member
  initialization to their target.

The two pieces mirror pattern 3 and share its machinery: a per-pass opt-in
table ``ConstructorFinalizationProfiles`` of the same
``FinalizationProfile<Node>`` row (here
``FinalizationProfile<CXXConstructorDecl>``), and a callback that emits via
``SemaProfiles::shouldEmitProfileViolation``.  The same shared
``dispatchFinalizationProfiles`` dispatcher invokes each callback, which
passes the ``CXXConstructorDecl`` to the decl-aware
``shouldEmitProfileViolation`` overload; that overload walks the declaration
and its lexical parents, so ``[[profiles::suppress]]`` on the constructor,
the class, or an enclosing lexical ``Decl`` works.  As for pattern 3, a
transient parse-time suppress scope belonging to an unrelated construct does
not reach these callbacks, because stack entries only match violations whose
tokens their construct covers (see :ref:`profiles-token-dominion`).  A
constructor body is normally instantiated lazily -- outside any unrelated
suppress scope -- so for pattern 4 this matters rarely; the scenario it
actually handles arises in pattern 3, where a class completes synchronously
inside an enclosing instantiation.  A callback that should only apply to
user-written constructors checks ``Ctor->isUserProvided()``.


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
(for example, the proposed ``[[uninit]]`` attribute of the
initialization profile) must introduce their own, separate, decl-scoped
marker.

This applies identically to the parse-time suppression stack and the
post-parse Stmt-tree walker described in pattern 2.

The parse-time stack enforces the dominion positionally: each entry records
the token range of the construct its attribute appertains to, and a
violation matches an entry only if its location falls within that range (in
translation-unit token order).  This is what keeps a live suppress scope
from leaking into code whose tokens it does not cover.  A check can fire
under an *unrelated* construct's scope in two ways: a template pattern
instantiated synchronously while the scope is live (instantiated code
retains the pattern's source locations, which lie outside the suppressed
construct wherever the pattern is declared -- before it, or first declared
after it), and a class or constructor finalized as a side effect of such an
instantiation (patterns 3 and 4).  In both cases the violation's location is
outside the entry's range, so the suppression correctly does not apply;
conversely, a local class or lambda *defined inside* the suppressed
construct is covered, whichever path re-enters it.

The range's end is recorded only when the construct was already fully
parsed when the entry was pushed -- a completed pattern or lexical parent at
an instantiation site, or a transformed ``AttributedStmt``.  For a construct
still being parsed no end is recorded (its end location would be
misleadingly early: a mid-parse class collapses to its name token, a
body-pending function to its declarator) and the entry's
``ProfileSuppressScope`` lifetime bounds the dominion instead.  That
fallback is exact mid-parse: the construct's later tokens do not exist yet,
and instantiation of a template that has no definition yet is deferred past
the scope's death.


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
during the appropriate region.  Each entry records the token range of the
construct its attribute appertains to (the end only once the construct is
fully parsed) and matches only violations located within it (see
:ref:`profiles-token-dominion`).  ``checkProfileViolation`` consults
``ProfileSuppressStack`` directly, so profile implementers never need to
create ``ProfileSuppressScope`` objects.

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

The reverse direction is *not* propagated: a suppress scope live at the
*point of instantiation* (for example on the declaration whose initializer
triggers it) covers the trigger's tokens, not the pattern's.  Instantiated
code retains the pattern's source locations, so the dominion check on stack
entries (see :ref:`profiles-token-dominion`) keeps such a scope from
suppressing checks that fire inside a synchronously instantiated body, NSDMI,
default argument, or instantiated marker re-check.

Module Enforcement
------------------

``[[profiles::enforce(...)]]`` on a module interface declaration records the
enforced profile designators on ``Module::EnforcedProfileDesignators``.  A
(non-partition) module implementation unit ``module M;`` automatically inherits
the interface's enforcements, because it implicitly imports the primary
interface unit of ``M``.
``[[profiles::require(...)]]`` on an import-declaration validates that the
imported module's ``EnforcedProfileDesignators`` contains a matching designator.

A *header unit* participates the same way: ``[[profiles::enforce(...)]]`` on
an empty-declaration in the header (the form P3589R2 §2.3 prescribes for
header units) is recorded on the header-unit module and serialized into its
BMI, so ``[[profiles::require]]`` on an ``import "header.h";`` validates
against it.  As with named modules, importing an enforced header unit does not
enforce the profile in the importer.

``[[profiles::enforce(...)]]`` on a *non-interface* module-declaration (a
``module M;`` implementation unit, or a ``module M:P;`` partition
implementation unit) is accepted but recorded only translation-unit-locally;
it is **not** added to ``Module::EnforcedProfileDesignators`` and so is not
visible to an importer's ``[[profiles::require]]``.

A module partition implementation unit ``module M:P;`` is also a module
implementation unit of ``M``, so the primary interface's enforcements apply to
it as well.  However, it does **not** implicitly import the primary interface,
and the primary interface is normally compiled *after* its partitions, so its
BMI is usually not available when the partition implementation unit is compiled.
Inheritance here is therefore **best-effort**: the enforcements are inherited
only when the primary interface's BMI is already resident in the compilation
(for example, supplied via an eager ``-fmodule-file=<path>``); the BMI is never
force-loaded and its absence is never diagnosed.  When it is not available the
partition implementation unit is simply not subject to the inherited profile --
a missed diagnostic, never a change to the meaning of a well-formed program.
For guaranteed enforcement, **repeat** ``[[profiles::enforce(...)]]`` in the
partition implementation unit rather than relying on inheritance.  (Best-effort
inheritance is silent when the interface BMI is absent; if the BMI *is* resident
and enforces a profile whose designator conflicts with a locally repeated
``enforce`` of the same name, that mismatch is still diagnosed.)

Importing a module that enforces a profile does **not** enforce that profile in
the importing translation unit.  Enforcement is always explicit and local.

Redeclaration Compatibility
---------------------------

P3589R2 [decl.attr.enforce]p5: a declaration and its redeclarations must
appear in the dominions of mutually compatible profiles.  The rule is
**symmetric** -- when a redeclaration is merged with a previous declaration
from another module unit, every profile whose dominion covered the previous
declaration must have a compatible counterpart covering the redeclaration,
and vice versa.  In particular, a profile-enforcing translation unit that
redeclares an entity from a module (or header unit) compiled *without* a
compatible profile is ill-formed; the paper's escape hatch for such headers
is ``[[profiles::exempt]]`` (not yet implemented, see `Intentional
Omissions`_).  The check
(``SemaProfiles::checkRedeclarationProfileCompatibility``) runs from
``Sema::CheckRedeclarationInModule``, the funnel for function, variable, tag,
alias, and class-template redeclarations; it is a framework rule -- a plain
error, not suppressible with ``[[profiles::suppress]]``, and diagnose-only
(the redeclaration still merges).

Two profiles are *compatible* if they have the same name -- designator
arguments configure a profile without changing its identity -- or if both are
standard (``std::``-prefixed) profiles, which P3589R2 proclaims mutually
compatible.  No further implementation-proclaimed compatibility is modeled.

The previous declaration's dominion is approximated by its top-level module's
exported ``EnforcedProfileDesignators``, which is exact for declarations in
the module purview -- including purview ``extern "C"``/``extern "C++"``
declarations (implicit global module), the common redeclarable case, since
module-attached entities cannot be redeclared in other translation units at
all.  Two cases have an *unknown* dominion and are skipped rather than
guessed at (a missed diagnostic, never a wrong one):

- A declaration in an **explicit global module fragment**: it precedes the
  module-declaration, so the exported enforcements do not cover it, and its
  TU's empty-declaration enforces are not serialized into the BMI.
- A previous declaration from the **same module family** (an implementation
  or partition unit merging with its own interface): the exported set
  under-approximates the interface TU's full dominion, and the interface's
  enforcements are inherited into the current unit anyway, so checking would
  false-positive on locally added profiles.

A textual or PCH previous declaration is not checked at all: it shares the
current TU's dominion (the placement rule makes a TU's dominion uniform, and
a PCH's enforcements are restored into the including TU).  Implicit template
instantiations are exempt, matching the module-ownership check.

Serialization
-------------

The framework serializes enforcement state automatically.  Profile implementers
do not need to add any serialization code.

- **PCH**: ``SemaProfiles::EnforcedProfiles`` is written as ``ENFORCED_PROFILES``
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


Built-in Profiles
=================

The tree ships five built-in profiles, all gated on ``-fprofiles``.  The four
``test::`` profiles are additionally gated on the ``-cc1``-only
``-fprofiles-test-profiles`` flag (see `Driver Flag`_) and are inert under
``-fprofiles`` alone; ``std::init`` needs only ``-fprofiles``:

- ``test::type_cast`` (test-only) -- pattern-1 example.
- ``test::uninit_read`` (test-only) -- pattern-2 example riding the existing
  CFG uninitialized-variables analysis.
- ``test::class_final`` (test-only) -- pattern-3 example riding the
  class-finalization dispatch.
- ``test::ctor_final`` (test-only) -- pattern-4 example riding the
  constructor-finalization dispatch.
- ``std::init`` (initial slice of the proposed initialization profile from
  Bjarne Stroustrup's "An initialization profile", P4222R1.1, on top of
  P3589R2 and P3402R3).  It uses all four patterns: the CFG dispatch (with
  ``test::uninit_read``), the constructor-finalization dispatch, and several
  parse-time check sites.  Paper section references (``§``) for ``std::init``
  in this document are to P4222R1.1.

By convention:

- Real test profiles live under the ``test::`` namespace.  Today there are
  four: ``test::type_cast``, ``test::uninit_read``, ``test::class_final``,
  and ``test::ctor_final``.  Because the ``test::`` prefix is what
  ``SemaProfiles::isProfileEnforced`` keys on to gate them behind
  ``-fprofiles-test-profiles``, any new test-only profile must also live
  under ``test::``.
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
  inside the ``reinterpret_cast`` arm.  Only the ``reinterpret_cast<>`` keyword
  form is checked; a C-style or functional cast with reinterpret semantics goes
  through a different path and is not diagnosed.

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
  guard consults it via ``hasEnforcedCFGUninitProfile()`` so the analysis
  runs even when ``-Wuninitialized`` is silenced, and
  ``UninitValsDiagReporter::diagnoseUnitializedVar`` walks it *before* the
  default warning path -- when an entry's
  ``SemaProfiles::shouldEmitProfileViolation`` returns true the entry's diagnostic
  fires and the default warning is skipped entirely.

The Stmt-tree suppression walker is what makes ``[[profiles::suppress]]``
work for this profile: by the time the CFG analysis runs, the parse-time
``ProfileSuppressStack`` has been unwound, so the helper consults the AST
directly via ``shouldEmitProfileViolation(name, rule, Stmt*, AnalysisDeclContext&)``.


The ``test::class_final`` Profile
---------------------------------

A pattern-3 (class-finalization) profile.  Demonstrates the case where the
rule applies once per completed class definition and runs from the
class-finalization dispatch in ``Sema::CheckCompletedCXXClass``.

- **Rules**: none (the profile has a single implicit rule, so the rule
  string is empty).
- **Diagnostic**: ``err_profile_class_final_test`` ("test profile fired on
  completion of class %1 under profile '%0'").
- **Opt-in table**: ``ClassFinalizationProfiles`` in
  ``clang/lib/Sema/SemaProfiles.cpp``.

Because dependent classes are filtered out by the dispatcher, the
diagnostic fires on class template *instantiations* rather than on the
primary template.  Lambda closures are also skipped.
``[[profiles::suppress(test::class_final)]]`` on the class or any
enclosing lexical ``Decl`` silences the diagnostic via the decl-aware
``shouldEmitProfileViolation`` overload, which walks the class and its
lexical parents for a matching suppression.


The ``test::ctor_final`` Profile
--------------------------------

A pattern-4 (constructor-finalization) profile.  Demonstrates the case where
the rule applies once per user-defined constructor, after its
member-initializer list is complete.

- **Rules**: none (single implicit rule, empty rule string).
- **Diagnostic**: ``err_profile_ctor_final_test`` ("test profile fired on
  finalization of a constructor for class %1 under profile '%0'").
- **Opt-in table**: ``ConstructorFinalizationProfiles`` in
  ``clang/lib/Sema/SemaProfiles.cpp``.

The diagnostic fires once per user-defined constructor -- written or implicit
member-initializer list, inline or out-of-line -- and on constructor template
*instantiations* rather than the dependent pattern.  Defaulted and implicit
constructors (no body) and delegating constructors are skipped.


The ``std::init`` Profile (initial slice)
-----------------------------------------

A slice of the proposed initialization profile.  A read of a scalar
``[[uninit]]`` data member before it is assigned *is* diagnosed by R1 via
CFG-based definite-assignment passes (paper §7.1 "initialized ... before
use"): over the constructor body for the current object's members, and over
any function body for the members of a *constructor-less aggregate local* --
the member-store slice of the paper §5.3 "classes exposing uninitialized
memory" pattern.  The rest of §5.3 (``construct_at``-based initialization),
random-access initialization of uninitialized arrays (paper §5.5), class-type
and array members (which need ``construct_at`` flow modeling),
``construct_at``/``destroy_at`` flow, and double-initialization /
double-destruction detection remain deferred.  The
constructor is *not* required to initialize a ``[[uninit]]`` member (paper §5.1
excepts members with an uninitialized indicator, and §5.3 leaves such a member
for the user): the obligation is keyed on a read before assignment, not on the
constructor's end.  A scalar *read through* a ``[[ref_to_uninit]]``
pointer/reference is diagnosed at the lvalue-to-rvalue conversion (R7,
``uninit_read``), and a member call on an object recognized as uninitialized
storage is diagnosed at its implicit object argument (R7, ``ref_to_uninit``);
*writes* through such a pointer/reference are not yet verified
(the paper relegates them to ``construct_at`` or suppression).  A scalar store
to a *subobject* of a named ``[[uninit]]`` entity, by contrast, is banned as
delayed piecemeal initialization (R10, ``uninit_write``).

Dynamically-created objects are covered when bound: a ``new`` expression that
default-initializes its allocated object and leaves a scalar subobject
indeterminate (e.g. ``new int``, ``new int[n]``, paper §1.2 / §4.3) is
recognised as a source of uninitialized memory by ``ref_to_uninit`` (R7).  An
*unbound* ``new`` whose result is discarded (``new int;``) is still unchecked,
because R7 fires only at binding sites.

The slice introduces two marker attributes and the rules below.

Marker attributes
~~~~~~~~~~~~~~~~~

``[[uninit]]`` (a standard C++11 attribute, distinct from the Clang
vendor attribute ``[[clang::uninitialized]]``) marks a ``VarDecl`` or
``FieldDecl`` as intentionally left uninitialized.  Recognised by Clang
regardless of ``-fprofiles``; its profile rules carry weight only when
``std::init`` is enforced.

- TableGen def: ``Uninit`` in ``clang/include/clang/Basic/Attr.td``,
  with a custom handler in ``clang/lib/Sema/SemaDeclAttr.cpp``.
- Subjects: ``Var`` and ``Field``.  The handler rejects placement where the
  marker is meaningless -- a reference, a function parameter, or a structured
  binding -- regardless of ``-fprofiles``.
- Behaviour:

  - Suppresses ``uninit_decl`` on the marked declaration (scalar or aggregate).
  - Excuses a non-static data member from ``ctor_uninit_member``.
  - Does **not** suppress ``uninit_read``.  Per the paper, the marker excuses
    the declaration but a read before any subsequent assignment is still
    ill-formed.
  - Triggers ``uninit_with_initializer`` when combined with an initializer,
    including a language-synthesized one from a constructor that actually runs
    (e.g. ``WithCtor x [[uninit]];``).  A trivial/aggregate type whose
    default-initialization is a no-op is *not* such an initializer, so the
    marker is accepted there (the object is genuinely left uninitialized).
  - Is banned on a pointer by ``pointer_marker`` (a pointer must be
    initialized, paper §4.3), and on a union object or member by
    ``union_marker`` (see R6 / R8 below for usage examples). Both are gated on
    enforcement, and the marker is retained after the diagnostic so
    ``uninit_decl`` does not re-diagnose the entity.
  - Is banned on a variable with static or thread storage duration by
    ``static_marker`` (such a variable is zero-initialized, so the marker
    contradicts paper §4.2; see R9 below).

``[[ref_to_uninit]]`` (also a standard C++11 attribute) marks a pointer,
reference, or pointer/reference-returning function as referring to
*uninitialized* memory.  Recognised by Clang regardless of ``-fprofiles``; its
profile rule carries weight only when ``std::init`` is enforced.

- TableGen def: ``RefToUninit`` in ``clang/include/clang/Basic/Attr.td``, with a
  custom handler in ``clang/lib/Sema/SemaDeclAttr.cpp``.
- Subjects: ``Var``, ``Field``, and ``Function``.  The handler rejects any
  subject whose type (or, for a function, return type) is not a pointer or
  reference to an object, via ``err_ref_to_uninit_attr_invalid_type`` --
  regardless of ``-fprofiles``.  A function pointer or reference (and a
  pointer-to-member) denotes a function or member, never uninitialized memory,
  so it is rejected too.
- Behaviour: drives the ``ref_to_uninit`` rule (below); has no other effect.

Rules
~~~~~

R1. ``uninit_read`` -- pattern 2 (CFG)
......................................

Reads of an uninitialized variable.  Implemented as a second row in the
existing ``CFGUninitProfiles`` table beside ``test::uninit_read``:

.. code-block:: c++

   constexpr CFGUninitProfileEntry CFGUninitProfiles[] = {
       {"test::uninit_read", /*Rule=*/"", diag::err_profile_uninit_read,
        /*ExemptStdByte=*/false},
       {"std::init", "uninit_read", diag::err_init_uninit_read,
        /*ExemptStdByte=*/true},
   };

If both ``test::uninit_read`` and ``std::init`` are enforced in the same TU,
the table-order priority makes ``test::uninit_read`` fire first.  Use
``[[profiles::suppress(test::uninit_read)]]`` to demote it at a use site
and surface the ``std::init`` diagnostic.

A read of an uninitialized ``std::byte`` is not diagnosed (paper §4.5 exempts
``std::byte``).  The exemption is per-entry via ``CFGUninitProfileEntry`` so it
applies to ``std::init`` but not the generic ``test::uninit_read`` profile.

The same ``std::init`` guarantee also covers a ``[[uninit]]`` scalar *data
member* read before it is assigned in a constructor body.
``checkInitProfileCtorBody`` in ``clang/lib/Sema/AnalysisBasedWarnings.cpp``
runs a forward definite-assignment dataflow over the constructor-body CFG: a
member is assigned by a plain ``m = e`` (for a built-in type a write is its
initialization, paper §4.5) and is *definitely assigned* at a point only if
assigned on every path reaching it (the meet is intersection, paper §1.3
"consider all branches ... executed").  A value read (an lvalue-to-rvalue load,
including the RHS of an assignment) of a member that is not yet definitely
assigned is reported via ``err_init_member_read_before_init`` at the first such
read, with a ``note_init_uninit_member_here`` note.  A compound assignment
``m op= e`` and a built-in increment or decrement (``++m``, ``m++``, ``--m``,
``m--``) read the member's old value before writing it, so each is treated as a
read-then-write of that member.  Details:

- It runs from ``IssueWarnings`` for an enforced ``std::init`` constructor,
  reusing the CFG built for the uninitialized-variables analysis, and also from
  the post-error path (``runUninitProfileAnalysisAfterError``) so an earlier TU
  error does not silently disable it.
- It reuses the ``uninit_read`` rule name (it enforces the same "no read of an
  uninitialized object" guarantee), so
  ``[[profiles::suppress(std::init, rule: "uninit_read")]]`` -- checked at the
  read site via the ``Stmt``/``AnalysisDeclContext`` suppression overload --
  covers it, and the ``std::byte`` exemption applies.
- Target members are ``[[uninit]]`` built-in scalar (arithmetic or enum)
  members; a member given a value by the *written* member-initializer list is
  assigned at its own initializer, in execution (declaration) order, so a later
  body read is fine (no spurious "marker + list-init" contradiction) while an
  *earlier* member initializer -- or a base initializer -- that reads it is a
  read-before-init (``X() : o(m) {}``).  An NSDMI's subexpressions are not
  expanded into the CFG, so a read of a tracked member inside another member's
  default initializer stays undetected.
- A member access on the current object is recognized whether spelled
  ``this->m``, implicitly as ``m``, or as the equivalent ``(*this).m``; an
  access through any other object (``other.m``) is not the current object's
  member.
- ``[[uninit]]`` members inherited from a non-virtual base with no
  user-provided constructor are tracked like the class's own members (nothing
  can have assigned them before the derived body runs); a written base
  initializer (``: Base{1}``) counts as assigning that base subtree's tracked
  members.  A base *with* a user-provided constructor is trusted (paper §5.1)
  -- its constructor body may have assigned the member, which this local
  analysis cannot see -- so its members are not tracked.
- A ``this``-capturing lambda created in the body may run immediately, so a
  member read in its body (or a nested lambda's) counts as a read at the point
  the lambda is created; a body write earns no assignment credit (the lambda
  may never run).  A lambda stored now but called only after the member is
  assigned is flagged all the same -- an accepted imprecision.  An
  init-capture's initializer runs at lambda creation and is checked as an
  ordinary read.
- There is **no** constructor-exit requirement: a ``[[uninit]]`` member that is
  simply never read is left as-is (paper §5.1/§5.3), exactly as R5 structurally
  excuses a marked member -- the two checks are complementary.
- A *delegating* constructor is skipped: its target initializes the members
  before the delegating body runs, so trusting the target (paper §5.1) avoids a
  false positive, matching how R5 skips delegating constructors.
- Out of scope here (deferred, conservative omissions, not extensions):
  class-type and array members, ``construct_at`` flow, and double-init/destroy
  detection.  Taking the address of a member or binding a reference to it is R7
  (``ref_to_uninit``) territory and is treated as neither a read nor an
  initialization by this pass.

The same guarantee covers an ``[[uninit]]`` scalar member of a
*constructor-less aggregate local* (the paper §5.3 "class exposing
uninitialized members" pattern used with a local:
``struct Agg { int m [[uninit]]; };`` and ``Agg a; a.m = 5;``).
``checkInitProfileLocalMembers`` in
``clang/lib/Sema/AnalysisBasedWarnings.cpp`` -- the local-variable analog of
the ctor-body pass, sharing its tracked-member filter, event/replay shape,
suppression lookup, and post-error rerun -- runs the same forward
definite-assignment dataflow over *every* function definition's CFG.  This is
the flow tracking that lets the parse-time read-through preset (R7) drop the
top-level member marker without losing the read-before-write diagnosis.
Details:

- Tracked pairs are (local variable, member): an automatic-storage,
  non-parameter, non-reference local whose class -- with any base subtree
  contributing tracked members -- has **no user-provided constructor** (one
  is trusted per paper §5.1: its body may have assigned the member, which
  local analysis cannot see) and whose declaration is the bare ``Agg a;``
  form.  Any written initialization (``Agg a{}``, ``= {}``, ``= Agg()``, a
  copy) gives every member a value and leaves nothing to track, as does a
  local that is itself ``[[uninit]]``-marked -- its subobject accesses are the
  parse-time read-through / ``uninit_write`` rules' territory.  A member of
  an anonymous struct or union is not tracked (its access is an
  ``IndirectFieldDecl`` chain, not a direct ``a.m``), consistent with the
  anonymous-aggregate skips in the ctor-body pass and R5; arrays of
  aggregates are likewise out of scope (element tracking is the deferred
  ``construct_at`` slice).
- A plain member store ``a.m = e`` assigns the member (§4.5); a compound
  assignment and a built-in ``++``/``--`` read the old value first and are a
  read-then-write, exactly as in the ctor-body pass.
- Soundness over completeness: **any** other appearance of the variable --
  ``&a``, ``&a.m``, a reference binding, passing ``a`` to any function
  (``construct_at``, ``memcpy``), a member call, a lambda capture --
  conservatively marks every tracked member assigned from that point (the
  address may be used to initialize the object), so no legal program is
  rejected.  A backward ``goto`` across the declaration re-default-initializes
  the object, which the gen-only dataflow cannot model -- a possible missed
  diagnostic, matching the ctor-body pass's accepted imprecision level.
- Objects reached through parameters, references, or other objects are not
  tracked; with the user-provided-constructor trust above this makes the
  remaining "``uu.y`` read through another object" case a deliberate,
  documented trust decision (see the read-through preset under R7).
- Reports reuse ``err_init_member_read_before_init`` under the shared
  ``uninit_read`` rule, so ``[[profiles::suppress(std::init, rule:
  "uninit_read")]]`` at the read site covers it and the ``std::byte``
  exemption applies (a ``std::byte`` member is never tracked).

R2. ``uninit_decl`` -- pattern 1
.................................

An automatic-storage variable definition whose default-initialization
leaves it (or a scalar subobject) indeterminate must either carry
``[[uninit]]`` or be initialized.  This covers a scalar / pointer /
enum with no initializer, and -- per paper §5.4 ("classes without
constructors") -- an aggregate or trivially-default-constructible class type
whose default-initialization leaves a scalar subobject indeterminate (e.g.
``struct S { int x; }; S s;``).  A class type with a user-provided default
constructor is trusted; static / thread storage duration is excluded
(zero-initialized by language rule -- a ``[[uninit]]`` marker on such a
variable is instead rejected by ``static_marker`` (R9)).

- Diagnostic: ``err_init_uninit_decl``.
- Check site: ``Sema::ActOnUninitializedDecl`` in
  ``clang/lib/Sema/SemaDecl.cpp``, which is only reached for declarations
  with no initializer (so braced or value initialization such as
  ``S s = {1};`` and ``S s{};`` is unaffected -- omitted aggregate members
  are value-initialized).
- The aggregate case uses ``SemaProfiles::defaultInitLeavesScalarIndeterminate``
  with ``HonorUninitMarkers=true``, which recurses through bases and members,
  trusts user-provided default constructors, and skips data members marked
  ``[[uninit]]`` (acknowledged uninitialized, paper §5.3). So a type
  whose only indeterminate scalars are all marked is trusted
  (e.g. ``struct A { int x [[uninit]]; }; A a;`` is accepted), while a
  mixed type still fires for its unmarked scalars.

R3. ``static_runtime_init`` -- pattern 1
.........................................

A non-local variable whose initializer is not a constant expression must
be ``constinit`` (which is already a hard error from the existing
``ConstInitAttr`` arm).  Without ``constinit``, the existing
``-Wglobal-constructors`` warning would fire (off by default); under
``std::init`` this rule promotes that to a profile error.

- Diagnostic: ``err_init_static_runtime_init``.
- Check site: in ``Sema::CheckCompleteVariableDeclaration``, in the
  constinit cascade, immediately before the ``-Wglobal-constructors`` arm
  (so the profile error takes precedence when both would fire).

R4. ``uninit_with_initializer`` -- pattern 1
............................................

``[[uninit]]`` and an initializer on the same declaration is a
contradiction (the marker means "no initialization here").

- Diagnostic: ``err_init_uninit_with_initializer``.
- Check site: ``SemaProfiles::checkInitProfileUninitWithInitializer``, shared by
  ``Sema::CheckCompleteVariableDeclaration`` (variables) and
  ``Sema::ActOnFinishCXXInClassMemberInitializer`` (data members with a
  default member initializer).
- A ``RecoveryExpr`` placeholder (from a failed initialization) is not a
  user-written initializer and does not trigger the rule.
- The "initializer" includes a language-synthesized one from a constructor
  that actually runs (e.g. ``WithCtor x [[uninit]];``), but *not* a
  no-op trivial/aggregate default-initialization, where the marker is
  consistent with the object being left uninitialized.
- Unlike R2/R5, this "no-op?" test calls
  ``defaultInitLeavesScalarIndeterminate`` with ``HonorUninitMarkers=false``
  (the *factual* answer): a type whose members are themselves marked still
  default-initializes to a no-op, so the variable marker stays consistent and
  the rule must not fire (e.g. ``A a [[uninit]];`` for the ``A`` above).

R5. ``ctor_uninit_member`` -- pattern 4
.......................................

A user-provided constructor must initialize every non-static data member
via its member-initializer list or an NSDMI, unless the member is marked
``[[uninit]]`` (paper §5.1).  A plain assignment in the constructor
body does not count.  A member whose own default-initialization leaves an
*unacknowledged* scalar subobject indeterminate (a nested aggregate) is
flagged as well; a member whose type's indeterminate scalars are all
themselves marked ``[[uninit]]`` is trusted (the same
``HonorUninitMarkers`` walk as R2, paper §5.3).  A direct non-virtual
base-class subobject left indeterminate is flagged the same way: the
guarantee is over the *complete object* (paper §5.1, §7.1), and -- unlike a
member -- a base cannot carry an ``[[uninit]]`` marker, so it must always be
initialized.  A written base-initializer (``: Base(...)`` / ``: Base{}``) or
a base with a user-provided default constructor is trusted.

- Diagnostic: ``err_init_ctor_uninit_member`` (with a
  ``note_init_uninit_member_here`` note at the member); for a base subobject,
  ``err_init_ctor_uninit_base`` (with a ``note_init_uninit_base_here`` note).
  Both share the ``ctor_uninit_member`` rule name, so one
  ``[[profiles::suppress(std::init, rule: "ctor_uninit_member")]]`` covers
  members and bases alike.
- Opt-in table: ``ConstructorFinalizationProfiles`` (pattern 4).
- Reference and const members keep their existing dedicated diagnostics;
  anonymous-aggregate members and unnamed bit-fields are skipped (named
  bit-fields are checked like any other member).
- A union's own constructor is exempt from this rule -- its members are
  mutually exclusive, so a constructor initializes at most one (paper §5.6;
  see R6). A union *data member* of a non-union class is still checked, and
  must be initialized via the member-initializer list.
- Known gaps: *virtual* base-class subobjects are not checked.  A virtual
  base is initialized by the most-derived constructor, which is not a local
  property of the constructor being checked, so flagging an intermediate
  constructor would push a redundant (and possibly surprising) ``: V()`` onto
  code that correctly relies on the most-derived class; under-diagnosing here
  is the safer, paper-consistent default.  Direct non-virtual bases are
  checked.  A const member is skipped here but is treated as indeterminate by
  ``defaultInitLeavesScalarIndeterminate`` (R2).

R6. ``union_marker`` -- attribute handler
.........................................

``[[uninit]]`` on a union object or a union member is banned (paper
§5.6): delayed initialization by assigning a member would be an erroneous
assignment when compiled without the profile.

.. code-block:: c++

   union U { int x; float y; };

   U a [[uninit]];   // error: [[uninit]] on a union variable (union_marker)
   U b;              // error: a union must be initialized (uninit_decl / err_init_uninit_union)
   U c = {1};        // OK
   U d{};            // OK

   union M {
     int x [[uninit]];  // error: [[uninit]] on a union member (union_marker)
     float y;
   };

   [[profiles::suppress(std::init)]] U e [[uninit]];  // OK

- Diagnostic: ``err_init_union_marker``.
- Check site: the shared helper ``SemaProfiles::checkInitProfileMarkerPlacement``,
  called from the ``Uninit`` handler in ``clang/lib/Sema/SemaDeclAttr.cpp`` and
  re-run on the instantiated entity from ``VisitFieldDecl`` / ``VisitVarDecl``
  in ``clang/lib/Sema/SemaTemplateInstantiateDecl.cpp``.  Unlike the reference /
  parameter / structured-binding rejections, which are unconditional, this is
  gated on enforcement -- a union may legitimately carry the marker without the
  profile.  Being Decl-aware it defers on a templated pattern and fires once on
  the instantiation (a dependent member or local that substitutes to a union),
  consistent with the other ``std::init`` rules.
- The rule keys on the *base element type*, so an array of unions
  (``[[uninit]] U a[2];``) is banned exactly like a single union object.  A
  union-typed data member of a non-union class is banned as well -- delayed
  initialization by assigning one of its members is just as erroneous there --
  in addition to the members *of* a union shown above.
- The banned marker is retained on the declaration after it is diagnosed, so
  the ``uninit_decl`` / ``ctor_uninit_member`` rules treat the entity as
  acknowledged and do not emit a second, contradictory diagnostic.  A member
  assignment to such a marker-retaining union (the §5.6 delayed-initialization
  ban) is caught by ``uninit_write`` (R10).

An *unmarked* union left uninitialized is itself the error (paper §5.6):
``SemaProfiles::defaultInitLeavesScalarIndeterminate`` reports a union as indeterminate
unless it has no members, a user-provided default constructor, or a default
member initializer.  A uninitialized union variable is therefore diagnosed by
``uninit_decl`` (with the union-specific ``err_init_uninit_union``) and an
uninitialized union data member by ``ctor_uninit_member``.

R7. ``ref_to_uninit`` -- pattern 1
..................................

A pointer or reference must be bound consistently with its
``[[ref_to_uninit]]`` marking (paper §4.3): a marked pointer/reference may only
refer to uninitialized memory, and an unmarked one may only refer to
initialized memory.  "Refers to uninitialized memory" is recognised purely
locally from the source expression's syntactic form (no flow analysis): the
address of, or a subobject of, a ``[[uninit]]`` entity; a value of a
``[[ref_to_uninit]]`` pointer/reference or array; a dereference of such a
pointer; a cast of such a pointer to another pointer type (paper §4.3), or of
such a glvalue to another reference; a call to a
``[[ref_to_uninit]]``-returning function; or a ``new`` expression that
default-initializes its allocated object and leaves a scalar subobject
indeterminate (e.g. ``new int``, ``new int[n]``, paper §1.2 / §4.3).
Pass-through forms are transparent to the operand they forward: a single-element
braced initializer (``{e}``) is looked through to its element, a conditional
(``c ? a : b``) is uninitialized if either arm is, and a comma (``(a, b)``)
takes its right operand -- so each is handled like the direct binding it
forwards.  A source whose form is recognized as neither uninitialized nor
trusted-initialized is classified as *unknown* rather than assumed initialized,
so it is diagnosed for neither a marked target (avoiding a false rejection) nor
an unmarked one (leaving a possible missed diagnostic).  The
reference cast and the ``[[ref_to_uninit]]``-returning reference call are not
spelled out by the paper but follow from the profile's guarantee that
uninitialized objects are not used, and keep the pointer and reference
recognizers symmetric.

- Diagnostics: ``err_init_ref_to_uninit_requires_uninit`` (marked target,
  initialized source) and ``err_init_uninit_requires_ref_to_uninit`` (unmarked
  target, uninitialized source).
- Recognizer + shared check: ``SemaProfiles::refersToUninitializedMemory`` and
  ``SemaProfiles::checkInitProfileRefToUninit`` in ``clang/lib/Sema/SemaProfiles.cpp``.
- A ``new`` expression is recognised only when it default-initializes its
  object (none init style, no written initializer); ``new T(...)`` and
  ``new T{...}`` are value- or list-initialized and excluded.  Whether the
  allocated type leaves a scalar indeterminate reuses
  ``SemaProfiles::defaultInitLeavesScalarIndeterminate`` (R2), so a ``T`` with a
  user-provided default constructor is trusted and the ``std::byte`` exemption
  is inherited.  ``getAllocatedType`` yields the element type, so array
  ``new`` (``new int[n]``) is handled uniformly.
- Check sites: variable initialization
  (``Sema::CheckCompleteVariableDeclaration``), default member initializers
  (``Sema::ActOnFinishCXXInClassMemberInitializer``), constructor
  member-initializers (``Sema::BuildMemberInitializer``), aggregate/list field
  initialization (``InitListChecker``), pointer assignment
  (``Sema::CreateBuiltinBinOp``), call arguments at parameter
  copy-initialization (``Sema::PerformCopyInitialization``, the funnel for
  every call form -- plain calls, constructor calls, overloaded operators,
  and calls to objects of class type such as functors and lambdas; a call
  with *no declared callee*, through a function pointer, is checked there
  too, its parameters treated as unmarked targets since no declaration could
  carry ``[[ref_to_uninit]]`` (paper §7.2) -- so passing uninitialized
  memory through a function pointer diagnoses even when the pointed-to
  function's own parameter is marked, the marker being a declaration
  property invisible through the pointer; suppress at the call if the flow
  is intended), arguments supplied by a parameter's default argument
  (``Sema::GatherArgumentsForCall``, which reuses the pre-built expression
  rather than re-running copy-initialization), variadic (``...``) arguments
  (the promotion loops in ``GatherArgumentsForCall`` and
  ``Sema::BuildCallToObjectOfClassType``, so variadic functors and variadic
  lambdas are covered too -- a ``...`` parameter cannot carry the marker, so
  a pointer argument is checked as an unmarked target, paper §7.2, while a
  promoted *value* read stays the read-through chokepoint's), return
  statements
  (``Sema::BuildReturnStmt``), implicit object arguments
  (``Sema::PerformImplicitObjectArgumentInitialization``, the funnel every
  member-call flavor's object argument converts through -- dot and arrow
  calls, member operators including whole-object ``operator=``, functor
  ``operator()``, ``operator->``, and conversion operators; the implicit
  object parameter can never carry ``[[ref_to_uninit]]``, so a call on an
  object recognized as uninitialized storage is checked as an unmarked
  target, paper §7.2, with suppress as the escape -- an *explicit* object
  member function instead initializes its object as an ordinary parameter,
  which the parameter site above already owns and whose parameter can carry
  the marker; a destructor call is skipped, destruction being the deferred
  destroy_at slice, and so is a static call operator, which -- like a static
  member call -- evaluates the object argument without using its value), and
  lambda captures -- an init-capture binds
  like a variable initialization when its variable is created
  (``Sema::createLambdaInitCaptureVarDecl``), and a plain by-reference capture
  of an entity denoting uninitialized storage (an ``[[uninit]]`` variable, or
  a ``[[ref_to_uninit]]`` reference) is checked when the closure is built
  (``Sema::BuildLambdaExpr``).  A capture cannot carry the marker, so only the
  unmarked-direction violation can fire there; a *copy* capture is not a
  binding -- it reads the variable in the enclosing function's CFG, which is
  the flow-based ``uninit_read`` pass's territory.  Inside a template the
  sites split by timing.  The Decl-carrying variable, data-member, and
  constructor member-initializer sites defer on the pattern (the
  ``D->isTemplated()`` check in ``shouldEmitProfileViolation``) and fire
  once, at instantiation, on the instantiated ``Decl``; the constructor site
  passes the enclosing constructor and is re-run by
  ``BuildMemberInitializer`` at instantiation, exactly like
  ``ctor_uninit_member``.  The Decl-less call-argument, assignment, return,
  aggregate-field, object-argument, and capture sites instead defer only when
  the source (for
  the capture, the captured variable's type; for pointer assignment, also
  the LHS) is *instantiation-dependent* -- such constructs are always rebuilt
  at instantiation, where the re-run ``Build*`` / ``InitListChecker`` routine
  checks the substituted form.  A non-dependent construct fires at
  *definition time* (TreeTransform can reuse it unchanged, so deferring
  would lose the diagnostic) and repeats if the construct is rebuilt at
  instantiation anyway; see Pattern 1 above for the unified model, its
  accepted phase-7 trade, and the accepted repetition.  The two timings are
  visible side by side:
  ``int *p = &g_uninit;`` in a template fires at instantiation (Decl-carrying
  variable site), while ``p = &g_uninit;`` fires at definition time
  (Decl-less assignment site) -- a deliberate asymmetry.  The aggregate-field
  hooks
  (``CheckSubElementType`` for a pointer field, ``CheckReferenceType`` for a
  reference field) are scoped to a member subobject (``EK_Member`` with a
  non-null parent), so the enclosing variable/argument/return is left to its own
  site, and a top-level member braced-initializer -- which the constructor or
  NSDMI site already checks -- is not diagnosed twice.
- Read-through enforcement (paper §4.5): a scalar *read* through a
  ``[[ref_to_uninit]]`` pointer or reference loads an uninitialized value and is
  diagnosed at the single lvalue-to-rvalue chokepoint
  (``Sema::DefaultLvalueConversion`` calling ``SemaProfiles::checkInitProfileReadThrough``),
  which by-value reads -- copy-initialization, by-value arguments, returns, and
  operator/condition operands -- all funnel through.  It reuses the recognizer
  with its read access preset (``UninitAccessOpts``), reports the shared rule
  ``uninit_read`` via ``err_init_uninit_read_through``, and exempts ``std::byte``
  (paper §4.5).  ``UninitAccessOpts`` carries two axes -- the top-level
  ``[[uninit]]`` drop and a ``[[ref_to_uninit]]`` trust flag -- whose presets
  distinguish a *binding* source (markers count everywhere), a value *read*
  (this rule), and a scalar *store* (``uninit_write``, R10, which shares the
  drop and additionally trusts the ``[[ref_to_uninit]]`` arms).
  The read preset drops the ``[[uninit]]`` marker only for the
  *top-level* named entity, whose direct reads are owned three ways: a
  directly named ``[[uninit]]`` object is flow-tracked by the CFG
  ``uninit_read`` pass, a current-object ``[[uninit]]`` member by the
  ctor-body pass, and an ``[[uninit]]`` member of a constructor-less
  aggregate local by the local-aggregate pass (all three credit assignments;
  see R1).  A marked member of an object with a *user-provided* constructor
  reached through any other object (``uu.y`` after ``Slot uu;``) is instead
  **trusted**, deliberately: the constructor's body may have assigned the
  member (the §5.2 pattern), which local analysis cannot see, so paper §5.1's
  trust-the-constructor principle applies and its reads are not diagnosed
  anywhere.  A *subobject* read of a named ``[[uninit]]`` object
  (``s.x``, ``o.agg.f``) or array (``a[0]``, ``*a``, ``s.a[i]``) is recognized
  and diagnosed here: neither flow pass tracks members or array elements, and
  subobject-wise delayed initialization of an ``[[uninit]]`` object is itself
  banned (paper §5.4/§5.5; only whole-object ``construct_at`` re-initializes,
  which is uniformly unmodeled), so no assignment could have given the
  subobject a value.  Being Decl-less, it defers only on an
  instantiation-dependent glvalue (rebuilt and re-checked at instantiation)
  and otherwise fires at definition time, repeating if the read is rebuilt
  at instantiation anyway (accepted).  An address-of (``&*p``), a reference
  binding, a discarded-value expression (``(void)*p``), and a write (``*p = 5``
  or ``s.x = 1``) apply no lvalue-to-rvalue conversion and so are not reads
  (``s.x = 1`` is instead banned as a subobject store by ``uninit_write``,
  R10).  A compound assignment (``*p += 1``) and a built-in ``++``/``--``
  also read the old value while building no lvalue-to-rvalue node; their
  loads are checked at the operator sites instead
  (``Sema::CheckAssignmentOperands`` for the non-shift compounds, the
  increment/decrement arm of ``Sema::CreateBuiltinUnaryOp`` for
  ``++``/``--``), while the shift compounds keep loading through the
  chokepoint via their LHS promotion and are excluded from the operator hook,
  so exactly one diagnostic fires.  A *class-type* copy from ``*p`` (copy-,
  direct-, braced-, argument-, or return-copy) never reaches the chokepoint --
  record glvalues undergo no lvalue-to-rvalue conversion -- but is caught all
  the same by the *binding* rule at the copy constructor's reference
  parameter, so the diagnostic is
  ``err_init_uninit_requires_ref_to_uninit`` rather than
  ``err_init_uninit_read_through``.  The escape is the paper's own (§7.2): a
  copy constructor declared with a ``[[ref_to_uninit]]`` reference parameter
  accepts the uninitialized source.  Because this is a binding, not a read,
  the ``std::byte`` exemption does not apply to a record that merely
  *contains* ``std::byte`` members.
- Known gaps: recognition is purely of the source's syntactic form, so a
  binding whose underlying operand is unrecognized -- pointer arithmetic, an
  integer-to-pointer cast, or the *result* of a call through a function
  pointer (no ``FunctionDecl`` to read a return marker from) -- is classified
  as *unknown* and diagnosed for neither direction.  A ``[[ref_to_uninit]]`` target
  therefore accepts it (rather than the earlier *false positive*), while an
  unmarked target also accepts it (a remaining missed diagnostic).  The
  pass-through forms above forward to such an operand without laundering it, so
  they inherit this gap rather than introducing one.  Only plain ``=`` pointer
  assignment is covered;
  compound assignment is not a binding and is skipped.  Aggregate field
  initialization is checked per scalar field, so an array-of-pointer (or
  array-of-reference) member is out of scope -- its elements are
  ``EK_ArrayElement`` and the ``[[ref_to_uninit]]`` marking lives on the field,
  not the element -- as is a pointer/reference member reached through an
  ``IndirectFieldDecl`` (a member of an anonymous struct/union), consistent with
  the scalar slice.  A member call through a pointer-to-member
  (``(s.*pmf)()``) resolves no ``CXXMethodDecl`` at the call and bypasses the
  object-argument conversion, so its object goes unchecked -- the
  pointer-to-member analog of the call-through-function-pointer gap.

R8. ``pointer_marker`` -- attribute handler
...........................................

``[[uninit]]`` on a pointer is banned (paper §4.3): "a reference cannot
be uninitialized.  The initialization profile requires the same for pointers."
A pointer must instead be initialized (e.g. to ``nullptr``).

.. code-block:: c++

   int *p;             // error: must be initialized or marked [[uninit]] (uninit_decl)
   int *q = nullptr;   // OK: the prescribed fix
   int *r [[uninit]];  // error: [[uninit]] cannot be applied to a pointer (pointer_marker)

   struct S {
     int *p [[uninit]];  // error: also fires on a pointer data member
   };

   // Opt out if genuinely required:
   [[profiles::suppress(std::init, rule: "pointer_marker")]] int *x [[uninit]];  // OK

- Diagnostic: ``err_init_uninit_pointer_marker``.
- Check site: ``SemaProfiles::checkInitProfileMarkerPlacement``, alongside
  ``union_marker`` (see R6 for the shared parse-time handler and the
  re-check on the instantiated field / variable).  Like that rule it is gated on
  enforcement -- a pointer may legitimately carry the marker without the profile
  -- and the marker is retained after the diagnostic so ``uninit_decl`` does not
  also fire.
- The check keys on the base element type, so an array of pointers
  (``[[uninit]] int *a[2];``) is banned exactly like a single pointer -- the
  marker cannot smuggle uninitialized pointers past ``uninit_decl``
  element-wise.
- A pointer parameter is rejected earlier and unconditionally (the marker is
  meaningless on a parameter); a pointer-to-member is not a pointer type and is
  out of scope.

R9. ``static_marker`` -- pattern 1
..................................

A variable with static or thread storage duration is zero-initialized by
language rule (paper §3), so it is an initialized object; marking it
``[[uninit]]`` contradicts paper §4.2 ("an initialized object marked
``[[uninit]]`` is an error").  A pointer must be initialized; a static is
*already* initialized.

.. code-block:: c++

   int glob;                      // OK: zero-initialized
   int glob2 [[uninit]];          // error: zero-initialized (static_marker)
   static int s [[uninit]];       // error: also on an explicit static
   thread_local int t [[uninit]]; // error: thread storage is zero-initialized too

   void f() {
     static int ls [[uninit]];    // error: also on a block-scope static
   }

   // Opt out if genuinely required:
   [[profiles::suppress(std::init, rule: "static_marker")]] int x [[uninit]];  // OK

- Diagnostic: ``err_init_uninit_static_marker`` (a ``%select`` distinguishes
  static from thread storage).
- Check site: ``Sema::ActOnUninitializedDecl`` in
  ``clang/lib/Sema/SemaDecl.cpp``, beside the ``uninit_decl`` (R2) check --
  *not* the ``Uninit`` attribute handler that hosts ``union_marker`` /
  ``pointer_marker``.  The decl site is reached only for a definition with no
  written initializer (so a non-defining ``extern`` declaration, handled
  earlier, is excluded), and passing the ``VarDecl`` to
  ``shouldEmitProfileViolation`` makes the rule fire on instantiations rather
  than template patterns, like every other Decl-based rule.
- Unlike ``uninit_decl``, ``std::byte`` is *not* exempt here: a static
  ``std::byte`` is zero-initialized (it cannot be left indeterminate the way an
  automatic one can), so the marker is still contradictory.
- Partition with ``uninit_with_initializer`` (R4): a static ``[[uninit]]`` with
  a real initializer -- an explicit one, or a constructor that actually runs
  (e.g. ``static WithCtor w [[uninit]];``) -- is R4's; ``static_marker`` covers
  only the zero-initialized, no-real-initializer case R4 treats as a consistent
  no-op (it reuses ``defaultInitLeavesScalarIndeterminate`` with
  ``HonorUninitMarkers=false``, R4's factual choice).  Exactly one diagnostic
  fires in every case.

R10. ``uninit_write`` -- pattern 1
..................................

A scalar store to a *proper subobject* of a named ``[[uninit]]`` entity is
banned delayed initialization (paper §1 "reading or writing uninitialized
memory is an error"; §5.4 member-wise, §5.5 random-access element, §5.6
union-member).  Writing the whole named entity is that entity's
initialization (paper §4.5: for a built-in type, a write is its
initialization) and stays legal -- the flow passes (R1) credit it -- as does
the §5.2 constructor-body pattern, whose member stores reach the current
object through ``this``.  Only whole-object ``construct_at`` could make a
piecemeal-initialized object good, and construct_at flow is uniformly
unmodeled, so no store below a marked entity can be part of a valid
initialization sequence.

.. code-block:: c++

   struct S { int x; int y; };

   void f() {
     S s [[uninit]];
     s.x = 1;   // error: writing a member of an [[uninit]] object (uninit_write)
     [[uninit]] int a[2];
     a[0] = 1;  // error: writing an element of an [[uninit]] object
     int v [[uninit]];
     v = 7;     // OK: the write initializes the whole entity
   }

   void g(int *p [[ref_to_uninit]]) {
     *p = 5;    // OK: a write through the marker is the pointee's
                // initialization (the deferred construct_at slice)
   }

- Diagnostic: ``err_init_uninit_subobject_write`` (a ``%select``
  distinguishes a member store from an element store).
- Recognizer: the shared classifier run with its *write* access preset
  (``UninitAccessOpts``, see R7): the top-level drop makes a whole-entity
  store legal, and ``TrustRefToUninit`` classifies storage reached through a
  ``[[ref_to_uninit]]`` pointer/reference (or returned by a marked function)
  as unknown, so a store through the marker is neither banned nor endorsed.
  The shared arms cover member chains, array elements (``a[i]``, ``*a``,
  member arrays -- on a named object or the current one), ``(&s)->x``, and
  the conditional/comma/braced pass-through bases.
- Check sites: ``Sema::CheckAssignmentOperands`` -- the funnel both simple
  and compound assignment converge on, so ``=`` and every ``op=`` are checked
  exactly once, and class-typed ``operator=`` (which diverts to overload
  resolution) never reaches it -- and the built-in increment/decrement arm of
  ``Sema::CreateBuiltinUnaryOp`` (overloaded class ``++``/``--`` never
  reaches it).  Both are Decl-less sites: they defer only on an
  instantiation-dependent store target -- always rebuilt and re-checked at
  instantiation -- and otherwise fire at definition time, repeating if the
  operator is rebuilt at instantiation anyway (see Pattern 1 and R7).
- A compound assignment or a built-in ``++``/``--`` also *reads* the old
  value, so on a subobject of a marked object the R7 read-through diagnostic
  fires alongside this rule's (the shift forms load through their LHS
  promotion, the rest through R7's operator-site hooks).
- ``std::byte`` stores are exempt (paper §4.5), matching every read-side
  rule.
- Whole-object assignment to a marked class object (``s = S{...}``) diverts
  to the overloaded ``operator=`` path and so never reaches this rule; it is
  caught instead by the ``ref_to_uninit`` object-argument check (R7) when the
  member ``operator=`` binds its implicit object parameter to the marked
  object.  Class-type writes remain uniformly deferred with construct_at.
- Known gaps: writes through ``[[ref_to_uninit]]`` are deliberately out of
  scope (for a scalar the write is the initialization; verifying class-type
  writes needs construct_at modeling).

Diagnostic suppression
~~~~~~~~~~~~~~~~~~~~~~

Every rule is suppressible per-site with
``[[profiles::suppress(std::init)]]`` (covers all rules) or
``[[profiles::suppress(std::init, rule: "rule_name")]]`` (rule-targeted).
The token-based-dominion limitation noted earlier applies: a suppress
attribute on a ``VarDecl`` covers only that declaration's tokens.

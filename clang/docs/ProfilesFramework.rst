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
   silenced, OR an ``S.anyProfileEnforced(Table)`` check (the shared
   ``Sema::anyProfileEnforced`` gate, also used by the finalization dispatch)
   into the existing pass guard.  The in-tree example's pass guard becomes
   ``S.anyProfileEnforced(CFGUninitProfiles) || !Diags.isIgnored(...)``.

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


Pattern 3: Class-Finalization Profile
-------------------------------------

Used when the rule applies to a class as a whole and needs to run once after
the class definition is complete -- for example, "every non-private field of
this class must satisfy property X" or "this class's field set must look
like Y."  ``test::class_final`` is the in-tree example.

The dispatch point is ``Sema::checkProfileViolationsAtClassFinalization``,
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
   ``ClassFinalizationProfiles`` in ``clang/lib/Sema/SemaDeclCXX.cpp`` is a
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
   patterns 3 and 4) checks ``anyProfileEnforced(Table)``, sets up a
   ``ProfileSuppressScope(S, RD, /*WalkLexicalParents=*/true)``, iterates the
   table, skips entries whose profile is not enforced, and invokes the
   callback.  Because the
   suppress scope is established by the dispatcher, the callback can use
   the location-based ``shouldEmitProfileViolation`` overload and have
   ``[[profiles::suppress]]`` on the class or any enclosing lexical
   ``Decl`` work correctly.

2. **Emit diagnostics from the callback via**
   ``Sema::shouldEmitProfileViolation``.  Each callback decides where on
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

The dispatch point is ``Sema::checkProfileViolationsAtConstructorFinalization``,
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
``Sema::shouldEmitProfileViolation``.  The same shared
``dispatchFinalizationProfiles`` dispatcher establishes a
``ProfileSuppressScope(S, Ctor, /*WalkLexicalParents=*/true)`` around each
callback, so ``[[profiles::suppress]]`` on the constructor, the class, or an
enclosing lexical ``Decl`` works.  A callback that should only apply to
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
enforced profile designators on ``Module::EnforcedProfileDesignators``.  A
(non-partition) module implementation unit ``module M;`` automatically inherits
the interface's enforcements, because it implicitly imports the primary
interface unit of ``M``.
``[[profiles::require(...)]]`` on an import-declaration validates that the
imported module's ``EnforcedProfileDesignators`` contains a matching designator.

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
partition implementation unit rather than relying on inheritance.

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


Built-in Profiles
=================

The tree ships five built-in profiles, all gated on ``-fprofiles``:

- ``test::type_cast`` (test-only) -- pattern-1 example.
- ``test::uninit_read`` (test-only) -- pattern-2 example riding the existing
  CFG uninitialized-variables analysis.
- ``test::class_final`` (test-only) -- pattern-3 example riding the
  class-finalization dispatch.
- ``test::ctor_final`` (test-only) -- pattern-4 example riding the
  constructor-finalization dispatch.
- ``std::init`` (initial slice of the proposed initialization profile from
  Stroustrup's draft, on top of P3589R2 and P3402R3).  It uses all four
  patterns: the CFG dispatch (with ``test::uninit_read``), the
  constructor-finalization dispatch, and several parse-time check sites.

By convention:

- Real test profiles live under the ``test::`` namespace.  Today there are
  four: ``test::type_cast``, ``test::uninit_read``, ``test::class_final``,
  and ``test::ctor_final``.
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
  guard consults it via ``S.anyProfileEnforced(CFGUninitProfiles)`` so the analysis
  runs even when ``-Wuninitialized`` is silenced, and
  ``UninitValsDiagReporter::diagnoseUnitializedVar`` walks it *before* the
  default warning path -- when an entry's
  ``Sema::shouldEmitProfileViolation`` returns true the entry's diagnostic
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
  ``clang/lib/Sema/SemaDeclCXX.cpp``.

Because dependent classes are filtered out by the dispatcher, the
diagnostic fires on class template *instantiations* rather than on the
primary template.  Lambda closures are also skipped.
``[[profiles::suppress(test::class_final)]]`` on the class or any
enclosing lexical ``Decl`` silences the diagnostic via the
``ProfileSuppressScope(*this, RD, /*WalkLexicalParents=*/true)`` the
dispatcher establishes around each callback.


The ``test::ctor_final`` Profile
--------------------------------

A pattern-4 (constructor-finalization) profile.  Demonstrates the case where
the rule applies once per user-defined constructor, after its
member-initializer list is complete.

- **Rules**: none (single implicit rule, empty rule string).
- **Diagnostic**: ``err_profile_ctor_final_test`` ("test profile fired on
  finalization of a constructor for class %1 under profile '%0'").
- **Opt-in table**: ``ConstructorFinalizationProfiles`` in
  ``clang/lib/Sema/SemaDeclCXX.cpp``.

The diagnostic fires once per user-defined constructor -- written or implicit
member-initializer list, inline or out-of-line -- and on constructor template
*instantiations* rather than the dependent pattern.  Defaulted and implicit
constructors (no body) and delegating constructors are skipped.


The ``std::init`` Profile (initial slice)
-----------------------------------------

A slice of the proposed initialization profile.  It does not yet implement
``[[ref_to_uninit]]`` (paper §5), classes that expose uninitialized memory to
users (paper §6.2), or random-access initialization of uninitialized arrays
(paper §6.4); and the constructor-body flow check that would let a
``[[uninitialized]]`` member be initialized by assignment in the body (the
dynamic half of paper §6.1) is deferred to a future CFG-based pass.  Until it
lands, a ``[[uninitialized]]`` data member is trusted.

The slice introduces one new attribute and the rules below.

Marker attribute
~~~~~~~~~~~~~~~~

``[[uninitialized]]`` (a standard C++11 attribute, distinct from the Clang
vendor attribute ``[[clang::uninitialized]]``) marks a ``VarDecl`` or
``FieldDecl`` as intentionally left uninitialized.  Recognised by Clang
regardless of ``-fprofiles``; its profile rules carry weight only when
``std::init`` is enforced.

- TableGen def: ``CXX11Uninitialized`` in ``clang/include/clang/Basic/Attr.td``,
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
    (e.g. ``WithCtor x [[uninitialized]];``).  A trivial/aggregate type whose
    default-initialization is a no-op is *not* such an initializer, so the
    marker is accepted there (the object is genuinely left uninitialized).
  - Is banned on a union object or union member by ``union_marker``.

Rules
~~~~~

R1. ``uninit_read`` -- pattern 2 (CFG)
......................................

Reads of an uninitialized variable.  Implemented as a second row in the
existing ``CFGUninitProfiles`` table beside ``test::uninit_read``:

.. code-block:: c++

   constexpr CFGUninitProfileEntry CFGUninitProfiles[] = {
       {"test::uninit_read", /*Rule=*/"", diag::err_profile_uninit_read},
       {"std::init",         "uninit_read", diag::err_init_uninit_read},
   };

If both ``test::uninit_read`` and ``std::init`` are enforced in the same TU,
the table-order priority makes ``test::uninit_read`` fire first.  Use
``[[profiles::suppress(test::uninit_read)]]`` to demote it at a use site
and surface the ``std::init`` diagnostic.

R2. ``uninit_decl`` -- pattern 1
.................................

An automatic-storage variable definition whose default-initialization
leaves it (or a scalar subobject) indeterminate must either carry
``[[uninitialized]]`` or be initialized.  This covers a scalar / pointer /
enum with no initializer, and -- per paper §6 ("classes without
constructors") -- an aggregate or trivially-default-constructible class type
whose default-initialization leaves a scalar subobject indeterminate (e.g.
``struct S { int x; }; S s;``).  A class type with a user-provided default
constructor is trusted; static / thread storage duration is excluded
(zero-initialized by language rule).

- Diagnostic: ``err_init_uninit_decl``.
- Check site: ``Sema::ActOnUninitializedDecl`` in
  ``clang/lib/Sema/SemaDecl.cpp``, which is only reached for declarations
  with no initializer (so braced or value initialization such as
  ``S s = {1};`` and ``S s{};`` is unaffected -- omitted aggregate members
  are value-initialized).
- The aggregate case uses ``Sema::defaultInitLeavesScalarIndeterminate``,
  which recurses through bases and members, trusts user-provided default
  constructors, and excludes unions.

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

``[[uninitialized]]`` and an initializer on the same declaration is a
contradiction (the marker means "no initialization here").

- Diagnostic: ``err_init_uninit_with_initializer``.
- Check site: ``Sema::checkInitProfileUninitWithInitializer``, shared by
  ``Sema::CheckCompleteVariableDeclaration`` (variables) and
  ``Sema::ActOnFinishCXXInClassMemberInitializer`` (data members with a
  default member initializer).
- A ``RecoveryExpr`` placeholder (from a failed initialization) is not a
  user-written initializer and does not trigger the rule.
- The "initializer" includes a language-synthesized one from a constructor
  that actually runs (e.g. ``WithCtor x [[uninitialized]];``), but *not* a
  no-op trivial/aggregate default-initialization, where the marker is
  consistent with the object being left uninitialized.

R5. ``ctor_uninit_member`` -- pattern 4
.......................................

A user-provided constructor must initialize every non-static data member
via its member-initializer list or an NSDMI, unless the member is marked
``[[uninitialized]]`` (paper §6.1).  A plain assignment in the constructor
body does not count.  A member whose own default-initialization leaves a
scalar subobject indeterminate (a nested aggregate) is flagged as well.

- Diagnostic: ``err_init_ctor_uninit_member`` (with a
  ``note_init_uninit_member_here`` note at the member).
- Opt-in table: ``ConstructorFinalizationProfiles`` (pattern 4).
- Reference and const members keep their existing dedicated diagnostics;
  anonymous-aggregate members and bit-fields are conservatively skipped.

R6. ``union_marker`` -- attribute handler
.........................................

``[[uninitialized]]`` on a union object or a union member is banned (paper
§6.5): delayed initialization by assigning a member would be an erroneous
assignment when compiled without the profile.

- Diagnostic: ``err_init_union_marker``.
- Check site: the ``CXX11Uninitialized`` handler in
  ``clang/lib/Sema/SemaDeclAttr.cpp``.  Unlike the reference / parameter /
  structured-binding rejections, which are unconditional, this is gated on
  enforcement -- a union may legitimately carry the marker without the
  profile.

Diagnostic suppression
~~~~~~~~~~~~~~~~~~~~~~

Every rule is suppressible per-site with
``[[profiles::suppress(std::init)]]`` (covers all rules) or
``[[profiles::suppress(std::init, rule: "rule_name")]]`` (rule-targeted).
The token-based-dominion limitation noted earlier applies: a suppress
attribute on a ``VarDecl`` covers only that declaration's tokens.


In-Tree Tests
=============

These tests collectively exercise the framework and the built-in
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
- ``clang/test/SemaCXX/safety-profile-class-final.cpp`` -- the
  ``test::class_final`` profile: end-to-end exercise of the
  class-finalization dispatch (pattern 3) including basic firing, class
  template instantiation, lambda skipping, suppression on the class and on
  enclosing lexical parents, SFINAE exclusion, and the
  without-``-fprofiles`` ignored path.
- ``clang/test/SemaCXX/safety-profile-init-read.cpp`` -- the ``std::init``
  profile's ``uninit_read`` rule.  Same ``-DCASE=N`` style as the
  ``test::uninit_read`` test; CASE=4 additionally enforces
  ``test::uninit_read`` to exercise the table-order priority.
- ``clang/test/SemaCXX/safety-profile-ctor-final.cpp`` -- the
  ``test::ctor_final`` profile: end-to-end exercise of the
  constructor-finalization dispatch (pattern 4) including written /
  no-list / out-of-line / instantiated constructors, the delegating and
  defaulted skips, suppression, and the without-``-fprofiles`` path.
- ``clang/test/SemaCXX/safety-profile-init-decl.cpp`` -- the ``std::init``
  profile's ``uninit_decl`` rule for scalars / pointers / enums: require an
  initializer or ``[[uninitialized]]``; statics / thread-locals are
  excluded; class types with a user-provided default constructor are
  trusted.
- ``clang/test/SemaCXX/safety-profile-init-aggregate.cpp`` -- the
  ``uninit_decl`` rule for aggregates / trivially-default-constructible
  class types whose default-init leaves a scalar subobject indeterminate
  (paper §6); braced and value initialization are accepted.
- ``clang/test/SemaCXX/safety-profile-init-static.cpp`` -- the ``std::init``
  profile's ``static_runtime_init`` rule: non-local vars need a
  constant initializer; locals / static-locals / thread-locals are
  excluded; ``constinit`` failures still produce the existing hard error
  regardless of ``-fprofiles``.
- ``clang/test/SemaCXX/safety-profile-init-with-initializer.cpp`` -- the
  ``std::init`` profile's ``uninit_with_initializer`` rule: every
  combination of ``[[uninitialized]]`` placement (prefix / postfix) with
  every initializer form (``= e``, ``{}``, ``(e)``), plus the
  synthesized-initializer and RecoveryExpr cases.
- ``clang/test/SemaCXX/safety-profile-init-field-marker.cpp`` -- placement
  of ``[[uninitialized]]`` on data members, the marker / NSDMI
  contradiction, and rejection on references, parameters, and structured
  bindings.
- ``clang/test/SemaCXX/safety-profile-init-ctor.cpp`` -- the ``std::init``
  profile's ``ctor_uninit_member`` rule: member-initializer-list / NSDMI /
  marker coverage, the nested-aggregate and body-assignment cases,
  out-of-line and instantiated constructors, and suppression.
- ``clang/test/SemaCXX/safety-profile-init-union.cpp`` -- the ``std::init``
  profile's ``union_marker`` rule banning the marker on a union object or
  union member.
- ``clang/test/PCH/cxx-profiles-enforce.cpp`` -- ``[[profiles::enforce]]``
  state survives PCH serialization round-trip.

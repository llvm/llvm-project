======================
C++ Profiles Framework
======================

.. contents::
   :depth: 2
   :local:


Introduction
============

The C++ Profiles framework (`P3589R2
<https://open-std.org/JTC1/SC22/WG21/docs/papers/2025/p3589r2.pdf>`_) lets a
translation unit opt into additional language restrictions called *profiles*.
A profile is a named set of rules enforced by the compiler, each formulated
to keep the program free of a certain class of problems -- for example, use
of uninitialized memory.  Three attributes control it:

- ``[[profiles::enforce(...)]]`` requests enforcement of one or more profiles
  for the translation unit.
- ``[[profiles::suppress(...)]]`` locally exempts a declaration or statement
  from an enforced profile, or from a single rule of it.
- ``[[profiles::require(...)]]`` on a module import verifies that the
  imported module advertises a profile.

Profiles do not change the meaning of well-formed programs with no undefined
behavior.  Their effects are conceptually applied only after translation
phase 7: a profile cannot change the outcome of overload resolution or
template instantiation, and it is not possible to SFINAE on a profile
violation.

Profile names are open-ended: standard (``std::``-prefixed),
implementation-defined, and third-party profiles are all requested with the
same syntax, and enforcing a profile the implementation does not know is not
an error -- it simply has no rules to enforce.  Clang currently implements
one real profile, an initial slice of the proposed ``std::init``
initialization profile (see `The std::init Profile (initial slice)`_).  The
feature is experimental: attribute spellings, rule names, and diagnostics may
change.


Usage
=====

The framework is gated on the C++-only ``-fprofiles`` flag, which defaults to
off:

.. code-block:: console

   clang++ -std=c++23 -fprofiles example.cpp

The attributes accept both the ``[[profiles::name(...)]]`` and the
``[[using profiles: name(...)]]`` spelling and always require an argument
clause; see the :doc:`AttributeReference` for the per-attribute reference.

Without ``-fprofiles`` the attributes are ignored with a warning, and their
argument clauses are not checked against the P3589R2 grammar -- like any
standard attribute the implementation does not act on, an arbitrary
balanced-token argument clause is accepted.  Code annotated for a
profiles-enabled build therefore still compiles (modulo the warning) with the
feature off, and no profile rule ever fires.


Enforcing Profiles
==================

``[[profiles::enforce(profile-designator-list)]]`` requests enforcement of
the named profiles for the whole translation unit.  It may appear only on an
*empty-declaration* that precedes every other declaration at translation-unit
scope, or on a *module-declaration* (see `Profiles and Modules`_):

.. code-block:: c++

   [[profiles::enforce(std::init)]];
   [[profiles::enforce(vendor::hardened(fortify: 3))]];  // designator arguments

   #include <my/lib.h>

   int main() { /* ... */ }

A *profile-designator* is a ``::``-qualified profile name, optionally
followed by a parenthesized argument list.  The arguments are not subject to
name lookup; their interpretation is up to the profile.  Repeating an
enforcement with the same designator is allowed and has no effect, but
requesting the same profile with a different designator is an error, as is an
enforcement placed after another declaration:

.. code-block:: c++

   [[profiles::enforce(vendor::hardened(fortify: 3))]];
   [[profiles::enforce(vendor::hardened(fortify: 3))]];  // OK: no effect
   [[profiles::enforce(vendor::hardened(fortify: 2))]];  // error: same profile,
                                                         // different designator
   int x;
   [[profiles::enforce(std::init)]];  // error: does not precede 'x'


Suppressing Enforcement
=======================

``[[profiles::suppress(profile-name)]]`` on a declaration or statement
exempts it from the named profile's rules.  An optional ``rule:`` argument
narrows the suppression to a single named rule, and an optional
``justification:`` argument (a string literal) records why the suppression is
there:

.. code-block:: c++

   [[profiles::enforce(std::init)]];

   void fill(char *buf, int n);

   int main() {
     [[profiles::suppress(std::init,
                          rule: "uninit_decl",
                          justification: "buffer is filled in by fill()")]]
     char buffer[1024];
     fill(buffer, 1024);
   }

A suppression covers exactly the tokens of the declaration or statement it
appertains to -- nothing more.  For a variable declaration that includes the
initializer, so violations inside the initializer are silenced; but the
variable is *not* marked as exempt at later uses, which appear in other
declarations or statements and are checked normally:

.. code-block:: c++

   [[profiles::suppress(std::init)]] int x;  // OK: uninit_decl suppressed
   int y = x;  // error: 'x' is read before initialization

To exempt an object from a profile's checks everywhere it is used, use the
profile's own per-object marker instead (for ``std::init``, ``[[uninit]]``).


Profiles and Modules
====================

A module interface advertises the profiles it enforces through
``[[profiles::enforce]]`` on its module-declaration, and importers can insist
on that advertisement with ``[[profiles::require]]``, which may appear only
on a module-import-declaration:

.. code-block:: c++

   // M.cppm
   export module M [[profiles::enforce(std::init)]];

   // user.cpp
   import M [[profiles::require(std::init)]];  // OK: M enforces std::init
   import N [[profiles::require(std::init)]];  // error unless N does too

``[[profiles::require]]`` only verifies the advertisement; importing an
enforcing module does **not** enforce its profiles in the importer.
Enforcement is always explicit and local.  A header unit participates the
same way: an ``[[profiles::enforce(...)]];`` empty-declaration in the header
is exported by the corresponding header unit and validated by
``[[profiles::require]]`` on its import.

Enforcement on a module interface extends to the module's implementation
units:

- A non-partition implementation unit (``module M;``) inherits the
  interface's enforcements automatically.
- A partition implementation unit (``module M:P;``) inherits them only on a
  best-effort basis, because the interface's BMI is usually not built yet
  when the partition is compiled.  Repeat the ``[[profiles::enforce]]`` there
  for guaranteed enforcement.

A declaration and its redeclarations must appear under mutually *compatible*
profiles (P3589R2 [decl.attr.enforce]p5): redeclaring an entity from a module
or header unit that was compiled without a compatible profile is diagnosed.
Two profiles are compatible when they have the same name (designator
arguments configure a profile without changing its identity), and all
standard ``std::`` profiles are mutually compatible.


The ``std::init`` Profile (initial slice)
=========================================

``std::init`` is an initial slice of the proposed initialization profile from
Bjarne Stroustrup's "An initialization profile" (P4222R1.1), on top of P3589R2
and P3402R3.  Paper section references (``§``) for ``std::init`` in this
document are to P4222R1.1.

A read of a scalar
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
-----------------

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
-----

R1. ``uninit_read`` -- pattern 2 (CFG)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
----------------------

Every rule is suppressible per-site with
``[[profiles::suppress(std::init)]]`` (covers all rules) or
``[[profiles::suppress(std::init, rule: "rule_name")]]`` (rule-targeted).
The token-based-dominion limitation noted earlier applies: a suppress
attribute on a ``VarDecl`` covers only that declaration's tokens.


Test Profiles
=============

Clang also ships four ``test::`` profiles (``test::type_cast``,
``test::uninit_read``, ``test::class_final``, and ``test::ctor_final``) that
exist only to exercise the framework in the test suite.  They are inert
without an additional ``-cc1``-only flag; see
:doc:`ProfilesFrameworkInternals`.


Not Yet Implemented
===================

``[[profiles::exempt(...)]]`` (P3589R2 §1.1.6), which would exempt named
included source files from the enforcement of a profile, is not implemented.


Extending the Framework
=======================

The framework is profile-agnostic: profile names are opaque strings, there is
no central registry, and adding a new profile requires no changes to the
framework itself.  See :doc:`ProfilesFrameworkInternals` for the
implementation patterns and the API for adding a new profile.

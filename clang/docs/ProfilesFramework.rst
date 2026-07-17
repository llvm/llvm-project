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
initialization profile (see `The std::init Profile`_).  The
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

``-fprofiles`` does not have to be uniform across a build: a module built
without the flag imports fine into a profiles-enabled compile -- it simply
advertises no profiles, so a ``[[profiles::require]]`` on the import reports
the profile as not enforced -- and an enforcing module loads fine into a
compile with the feature off.  A PCH is stricter (like other compatible
language options, it must be built with the same ``-fprofiles`` setting as
its consumer).

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


The ``std::init`` Profile
=========================

``std::init`` is an initial slice of the proposed initialization profile from
Bjarne Stroustrup's "An initialization profile" (P4222R1.1; the ``§``
references below are to that paper).  Its guarantee: **no object is read or
written before it is initialized**, enforced entirely at compile time.
Following the paper:

- Every object must be initialized at its point of definition, or be marked
  as intentionally uninitialized with the ``[[uninit]]`` attribute.
- An object marked ``[[uninit]]`` must be assigned before it is read, which
  is verified with simple, local flow analysis (§1.3).
- A pointer or reference to uninitialized memory must be marked with the
  ``[[ref_to_uninit]]`` attribute, and reads through it are rejected.
- Non-local objects must be initialized at compile time (§3).
- Implicit initialization counts: a default constructor or the
  zero-initialization of statics initializes an object, and a class with a
  user-provided constructor is trusted to initialize its members (§5.1).

``[[uninit]]`` and ``[[ref_to_uninit]]`` are ordinary C++11 attributes,
recognized regardless of ``-fprofiles`` (see the :doc:`AttributeReference`
entries for their placement rules); the rules below carry weight only while
``std::init`` is enforced.  Reads and writes of ``std::byte`` objects are
exempt from all of them (§4.5).

Each rule has a name, so it can be suppressed individually with
``[[profiles::suppress(std::init, rule: "name")]]`` (see `Suppressing
Enforcement`_):

=========================== ======================================================
Rule                        Diagnoses
=========================== ======================================================
``uninit_decl``             A variable left (partially) uninitialized without
                            ``[[uninit]]``.
``uninit_read``             A read of an uninitialized object, or of an
                            ``[[uninit]]`` object before it is assigned.
``uninit_write``            A write to a subobject of an ``[[uninit]]`` object.
``ref_to_uninit``           A pointer or reference binding inconsistent with its
                            ``[[ref_to_uninit]]`` marking.
``ctor_uninit_member``      A constructor that leaves a member or base subobject
                            uninitialized.
``static_runtime_init``     A non-local variable with a runtime initializer.
``uninit_with_initializer`` ``[[uninit]]`` combined with an initializer, or
                            on an entity whose default-initialization is not
                            a no-op.
``pointer_marker``          ``[[uninit]]`` on a pointer.
``union_marker``            ``[[uninit]]`` on a union object or member.
``static_marker``           ``[[uninit]]`` on a variable with static or thread
                            storage duration.
=========================== ======================================================


Uninitialized Variables
-----------------------

An automatic-storage variable whose default-initialization would leave it --
or, for an aggregate, any scalar subobject (§5.4) -- indeterminate must
either be initialized or carry ``[[uninit]]`` (rule ``uninit_decl``):

.. code-block:: c++

   struct S { int x; };
   union U { int i; float f; };

   void f() {
     int a;             // error: uninitialized (uninit_decl)
     int b [[uninit]];  // OK: intentionally uninitialized
     int c = 3;         // OK
     S s;               // error: default-initialization leaves 's.x' indeterminate
     S t{};             // OK: value-initialized
     U u;               // error: an uninitialized union (§5.6)
     std::string str;   // OK: a user-provided default constructor is trusted
   }

A class type with a user-provided default constructor is trusted to
initialize its members (§5.1), and a data member that is itself marked
``[[uninit]]`` is acknowledged -- a type whose only indeterminate scalars are
all marked does not trigger the rule.  Marking a variable of such a type
``[[uninit]]`` is likewise consistent: its default-initialization is a
genuine no-op.


Where ``[[uninit]]`` May Not Go
-------------------------------

``[[uninit]]`` asserts that an object is genuinely uninitialized, so
placements that contradict that -- or that would defeat the profile's
guarantee -- are rejected:

.. code-block:: c++

   int g [[uninit]];    // error: statics are zero-initialized (static_marker)
   int *p [[uninit]];   // error: a pointer must be initialized, e.g. to
                        // nullptr (pointer_marker, §4.3)

   union U { int i; float f; };
   U u [[uninit]];      // error: delayed initialization of a union member
                        // would be erroneous (union_marker, §5.6)

   void f() {
     int x [[uninit]] = 4;      // error: marked *and* initialized
                                // (uninit_with_initializer, §4.2)
     std::string s [[uninit]];  // error: the default constructor initializes
                                // it (uninit_with_initializer)
   }

Each of these keys on the array element type, so an array of pointers or of
unions is rejected exactly like a single one.  A no-op initialization -- the
trivial default-initialization of a scalar or aggregate that leaves a scalar
subobject indeterminate -- is consistent with the marker; any other
synthesized initialization contradicts it and is rejected (§5.3): a member or
base with a user-provided default constructor, a default member initializer,
a virtual table pointer, or a value-initialization (``= P()``), all of which
initialize something.  The same rule covers a marked data member whose type's
default-initialization is not a no-op:

.. code-block:: c++

   struct Str { Str() : cap(0) {} int cap; };
   struct S { int x; Str s; };

   void g() {
     S s4 [[uninit]];   // error: 's4.s' is default-constructed, so 's4' is
                        // not left uninitialized (uninit_with_initializer, §5.3)
   }

   struct Buf {
     Str s [[uninit]];  // error: default-initialization of 'Str' runs a
                        // constructor, so 's' cannot be left uninitialized
     int n [[uninit]];  // OK: a scalar member really is left uninitialized
   };


Reads of Uninitialized Objects
------------------------------

An uninitialized object must not be read (rule ``uninit_read``).  Local flow
analysis verifies reads of uninitialized locals, of ``[[uninit]]`` members
within the defining constructor's body, and of ``[[uninit]]`` members of
constructor-less aggregate locals; reads through ``[[ref_to_uninit]]``
pointers and references and reads of subobjects of ``[[uninit]]`` objects are
rejected outright -- unless a prior whole-entity store credits the storage
as initialized (see `Binding Pointers and References`_):

.. code-block:: c++

   struct Buf { int n [[uninit]]; };

   int f(int *p [[ref_to_uninit]]) {
     int x [[uninit]];
     int a = x;    // error: 'x' is read before it is assigned
     x = 3;
     int b = x;    // OK: assigned on every path reaching the read

     Buf buf;
     int c = buf.n;  // error: 'buf.n' is not yet assigned
     buf.n = 1;
     int d = buf.n;  // OK

     [[uninit]] int arr[4];
     int e = arr[0]; // error: array elements are not tracked; only a
                     // whole-object initialization can give 'arr' a value

     return *p;      // error: read through [[ref_to_uninit]] (§4.5)
   }

   struct T {
     int m [[uninit]];
     T(int v) {
       int r = m;    // error: 'm' is read before the body assigns it
       m = v;        // OK: for a built-in type, a write is its
     }               // initialization (§4.5)
   };

A member or variable counts as assigned only when every path to the read
assigns it (§1.3), and a compound assignment (``x += 1``) or an increment or
decrement reads the old value first, so it is diagnosed like a read.  Inside
a constructor body only that plain whole-member assignment earns credit:
passing ``&m`` to a function (even one whose parameter is marked
``[[ref_to_uninit]]``), binding a reference to the member, calling a member
function, or letting ``this`` escape does not count as initializing ``m`` --
the paper rejects complex constructor code (§5.1) and reserves
callee-initialization for ``now_init()`` (§6.2); suppress the rule where
such a flow is intended.  A ``this``-capturing lambda might run immediately,
so member reads in its body count at the point the lambda is created (and
writes there earn no credit -- the same strict policy).  For a *local*
variable the analyses instead treat any escape as an assignment (see
`Limitations`_).  Members of an object initialized by a *user-provided*
constructor are trusted (§5.1) and not flow-tracked; only the defining
constructor itself is checked.


Writes to Subobjects of Uninitialized Objects
---------------------------------------------

Piecemeal delayed initialization of an ``[[uninit]]`` object through its
members or elements cannot be validated statically (§5.4, §5.5), so a store
to a *proper subobject* of an ``[[uninit]]`` entity is rejected (rule
``uninit_write``); writing the whole entity is that entity's initialization,
stays legal, and credits the entity as initialized for everything after it
in parse order (see `Binding Pointers and References`_):

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
                // initialization
   }

A compound assignment or increment/decrement of such a subobject also reads
its old value, so the read diagnostic fires alongside this one.  Assigning a
whole *class* object marked ``[[uninit]]`` (``s = S{1, 2};``) is caught too:
the member ``operator=`` binds the uninitialized object as its implicit
object argument (see the next section).


Binding Pointers and References
-------------------------------

A pointer or reference must be bound consistently with its
``[[ref_to_uninit]]`` marking (rule ``ref_to_uninit``, §4.3): a marked
pointer, reference, or function return may only refer to uninitialized
memory, and an unmarked one only to initialized memory.  Whether a source
refers to uninitialized memory is recognized from its form -- the address or
a subobject of an ``[[uninit]]`` entity, the value or dereference of a
marked pointer or reference, pointer and reference casts of those, a call to
a marked function, a call to a known allocator (``malloc``,
``aligned_alloc``, ``alloca``, and raw ``::operator new`` calls return
uninitialized memory, ``calloc`` zero-initialized memory, and ``realloc`` is
unclassified; §4.3 -- keyed on Clang's builtin recognition, which
``-fno-builtin`` disables), and a ``new``
expression that default-initializes a type with indeterminate scalars
(``new int``, ``new int[n]``; §1.2) -- refined by one parse-order fact,
whole-entity stores (below):

.. code-block:: c++

   int i = 7;
   int u [[uninit]];

   int *p1 = &i;                         // OK
   int *p2 = &u;                         // error: needs [[ref_to_uninit]]
   int *p3 [[ref_to_uninit]] = &u;       // OK
   int *p4 [[ref_to_uninit]] = &i;       // error: 'i' is initialized
   int *p5 [[ref_to_uninit]] = new int;  // OK: uninitialized new (§1.2)

   void sink(int *q);
   void fill(int *q [[ref_to_uninit]]);

   void f() {
     sink(p3);  // error: 'sink' expects initialized memory
     fill(p3);  // OK
   }

The check applies wherever a pointer or reference is bound: variable, member,
and aggregate initialization, assignment, call arguments (including defaulted
and variadic ones), ``return`` and ``throw`` statements (including returns
inside lambdas and blocks -- a lambda's marker is written on its call
operator, in the C++23 attribute position after the lambda-introducer:
``[] [[ref_to_uninit]] () -> int* { ... }``), lambda captures, and
the implicit object argument of a member call -- so calling a member function
on an object recognized as uninitialized storage is rejected, and so is
copying a class object out of one.  For the copy, the escape is the paper's
own (§7.2): declare the copy constructor's parameter ``[[ref_to_uninit]]``.
Positions that cannot carry the marker -- a variadic argument, a parameter of
a function called through a function pointer, the implicit object parameter,
a pointer element of an array in aggregate initialization
-- are checked as unmarked targets; suppress at the call site if the flow is
intended.  A null pointer source -- ``nullptr``, ``0``, ``{}``, or a local
variable initialized to null -- refers to no object, so it is accepted for
marked and unmarked targets alike (§4.3, §8: the marker means "zero or more
uninitialized objects"); a *parameter* with a null default argument is not a
null source (callers may pass any pointer).  A source whose form the recognizer cannot classify
(pointer arithmetic, an integer-to-pointer cast) is likewise accepted for
either target.

The parse-order refinement: a *whole-entity store* credits its target as
initialized (§4.2, §4.5).  After ``u = 5;`` the ``[[uninit]]`` variable
``u`` counts as initialized, and after ``*p = 5;`` the marked pointer's
pointee does, for every later ``*p`` access -- until ``p`` is reseated
(``p = q``, ``p += n``, ``p++``), which withdraws the credit; a store
through a marked *reference* credits its referent permanently.  The credit
works in both directions: ``int *q = &u;`` after the store is accepted, and
``int *r [[ref_to_uninit]] = &u;`` is now rejected -- a credited entity
requires an unmarked target (§4.2).  Element accesses (``p[i]``) are never
credited in either direction (§5.4's random-access ban applies even to
``p[0]``), element stores earn no credit, and address escapes (passing
``&u`` to a ``[[ref_to_uninit]]`` parameter) never count -- the paper
reserves callee-initialization for ``now_init()`` (§6.2).  Class-typed
whole-object assignment never credits either: it is a member ``operator=``
call on uninitialized storage, rejected as above.

Whole-member stores are credited too, keyed per base object: after
``a.m = 5;`` on a directly named local, or ``this->m = 5;`` on the current
object (keyed to the enclosing function body, so no other function -- nor a
lambda body, which is its own function -- shares it; instantiations of one
template or generic lambda share their pattern's single body, and its
credit), binding ``a.m`` or
``&a.m`` through the *same* base is accepted, and the reverse direction
applies as for locals.  The boundaries: one member level only (``x.agg.m =
5;`` is itself the piecemeal-initialization error and earns nothing), only
directly named non-reference locals or the current object (a member reached
through a pointer, reference, or any other object -- including a copy, per
§5.2 -- stays strict), and never member *pointee* stores (``*w.p = 5;``),
whose aliasing is per-value: a copy of the object shares the pointee.  The
credit is purely parse-order, with no dominance or flow analysis: a store
under a condition credits everything after it, so a binding on the untaken
path is a missed diagnostic (see `Limitations`_).

One kind of call *does* count as initialization: §6.2's ``[[now_init]]``
attribute (its placement and spelling track an open committee question)
declares that a function initializes the storage passed to each of its
``[[ref_to_uninit]]`` parameters, and it requires at least one such
parameter.  After a call to a ``[[now_init]]`` function, the argument's
storage earns exactly the credit the equivalent direct store would --
``fill(&u)`` credits ``u`` whole, ``fill(p)`` credits the marked pointer's
pointee (until ``p`` is reseated), ``fill(&a.m)`` credits the ``(a, m)``
pair -- with the same boundaries and the same reverse-direction consequence
(a second ``fill(&u)`` is rejected: ``u`` no longer refers to uninitialized
memory, which incidentally catches double ``construct_at``).  This is R2
§4.5's requested library annotation: declare ``construct_at`` with
``[[now_init]]`` and a ``[[ref_to_uninit]]`` first parameter and the
lifecycle *start* is checked.  The §4.4 ``now_init()`` identity function
needs no compiler support at all -- declared as ``template<class T> T*
now_init(T* p [[ref_to_uninit]]);``, its unmarked return is already trusted
as initialized -- but only the attribute legalizes the original *name* after
the call.


Constructors
------------

A user-provided constructor must initialize every non-static data member
through its member-initializer list or a default member initializer unless
the member is marked ``[[uninit]]`` (rule ``ctor_uninit_member``, §5.1); an
assignment in the constructor body does not count (reads of an ``[[uninit]]``
member before such an assignment are flow-checked, as above).  Direct
non-virtual base subobjects must be initialized the same way -- a base cannot
carry ``[[uninit]]``, so it must always be initialized, by a written base
initializer or by the base's own user-provided default constructor:

.. code-block:: c++

   struct X {
     int a;
     int b = 0;
     int c [[uninit]];
     X(int v) : a(v) {}  // OK: 'a' is written, 'b' has a default member
   };                    // initializer, 'c' is acknowledged

   struct Y {
     int m;
     Y() {     // error: 'm' is not initialized (ctor_uninit_member)
       m = 1;  //   (a body assignment does not count)
     }
   };

A member whose type's default-initialization leaves unacknowledged scalars
indeterminate (a nested aggregate) is flagged the same way.  A delegating
constructor is exempt -- its target initializes the members -- and so is a
union's own constructor, whose members are mutually exclusive (§5.6).


Global and Static Variables
---------------------------

Variables with static or thread storage duration are zero-initialized, so
they are never uninitialized: ``[[uninit]]`` on one is rejected -- by
``static_marker`` when nothing else initializes the object, and by
``uninit_with_initializer`` when a written initializer or a non-no-op
default-initialization already contradicts the marker (exactly one of the
pair fires).  Non-local variables with static storage duration must
additionally be initialized at compile time (rule ``static_runtime_init``,
§3), because cross-translation-unit initialization order can otherwise
produce a read of a not-yet-initialized object:

.. code-block:: c++

   int seed();

   int g1 = 42;                  // OK: constant-initialized
   int g2 = seed();              // error: runtime initializer (static_runtime_init)
   thread_local int t = seed();  // OK: thread storage duration

   int &counter() {
     static int c = seed();  // OK: a function-local static is initialized
     return c;               // on first use (§3)
   }


Limitations
-----------

The implemented slice is deliberately conservative.  The first entry below
is a deliberate strictness -- it rejects code the paper itself rejects --
rather than an omission; each of the others is a missed diagnostic, never a
false positive.

- Inside a constructor body, only a plain assignment to an ``[[uninit]]``
  member or a call to a ``[[now_init]]`` function (§6.2) counts as its
  initialization -- the latter for the current-object storage bound to the
  callee's ``[[ref_to_uninit]]`` parameters (``&m``, ``m``, or ``this``
  itself, which credits every tracked member), as a genuine dataflow fact,
  so a ``[[now_init]]`` call on one branch still does not satisfy a read at
  the join (§1.2).  Taking the member's address, binding a reference to it,
  calling a member function, letting ``this`` escape, or passing ``&m`` to a
  ``[[ref_to_uninit]]`` parameter of an *ordinary* function earns no credit,
  so a later read of the member is rejected: the paper rejects complex
  constructor code (§5.1) and reserves callee-initialization for
  ``now_init`` (§6.2); the remedy for an intended flow is ``[[now_init]]``
  on the callee or ``[[profiles::suppress]]``.  For *locals*, by contrast,
  the local-aggregate pass and the plain-local analysis conservatively treat
  any escape of the variable as an assignment -- there the omission is a
  missed diagnostic, never a false positive.  That escape-crediting is an
  interim leniency relative to the paper (which credits only ``now_init``);
  tightening it to ``[[now_init]]`` callees alone is future work.
- ``construct_at``/``destroy_at`` flow is only partially modeled: a
  ``[[now_init]]``-annotated ``construct_at`` declaration checks the
  lifecycle start (including double construction, via the reverse-direction
  binding rule), but destruction -- ``destroy_at``, double destruction,
  use-after-destroy -- is not checked, and writes through
  ``[[ref_to_uninit]]`` are not verified.
- A ``new`` expression whose result is not bound to anything (``new int;``)
  is not checked.
- A call through a function pointer cannot see parameter markers on the
  pointed-to function, and a member call through a pointer-to-member bypasses
  the object-argument check.
- Members of anonymous structs and unions, and arrays of aggregates, are not
  flow-tracked.
- Virtual base subobjects are not checked by ``ctor_uninit_member`` (they are
  initialized by the most-derived class).
- A read of a tracked member inside another member's default initializer is
  not detected.
- Whole-entity store credit is parse-order only: a store under a condition
  (or inside a lambda body) credits every later use in parse order, so a
  read or binding on a path that skips the store is a missed diagnostic.
  ``[[now_init]]`` call credit outside constructor bodies is parse-order in
  the same way (inside them it is a real dataflow fact, as above).  The
  requires-uninitialized direction consults this credit at definition time
  only -- an instantiation re-walk does not rewind parse-order state, so
  re-checked statements must not trip over credit they themselves recorded
  -- which makes a reverse-direction violation established only by credit
  in fully dependent code a missed diagnostic as well.
- In a template, a violation in non-dependent code is diagnosed at definition
  time and may be repeated at instantiation.


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

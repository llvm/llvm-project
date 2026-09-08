.. _herbceptions:

==========================================================
Herbceptions
==========================================================

.. contents::
   :local:

.. note::

   This is an **experimental** extension under active development. The
   implementation is incomplete, the ABI is not stable, and the design may
   change. Enable it with ``-fherbceptions``.

Overview
========

Herbceptions (after Herb Sutter's `P0709R4 <https://wg21.link/p0709r4>`_) replace the traditional two-phase
C++ exception model with a **deterministic error channel**. A function that
can fail declares *that it can fail* with a ``throws`` (C++) or ``return_failure{
  E}``
(C and C++) specifier, and the error is returned through the normal return
path, discriminated by a boolean flag carried next to the value.

The two models are orthogonal:

* **Traditional EH** (``throw`` / ``try`` / ``catch``): requires
  ``-fexceptions`` and uses the unwinder, landing pads, and a personality
  function. The unwind is asynchronous; destructors run during phase-2
  unwinding.
* **Herbceptions** (``throw throws`` / ``try(expr)`` / ``catch return_failure(expr)``):
  enabled by ``-fherbceptions``. Errors are returned like ordinary values;
  there is no unwinder, no landing pad, no personality function, and no
  LSDA. Destructors run on the normal scope-exit path.

The two flags are independent: ``-fherbceptions`` controls the deterministic
error channel and does *not* require ``-fexceptions``.

Why not traditional exceptions?
-------------------------------

Traditional C++ exceptions were rejected as the basis of this design because
of the implementation and runtime costs that come with a two-phase,
stack-unwinding model:

* **Implementation difficulty.** A correct two-phase unwinder (search phase,
  then unwind phase), the LSDA encodings, and the personality functions
  (Itanium ``__gxx_personality_v0``, MSVC ``__CxxFrameHandler3``, Wasm) are
  large, fragile systems. A deterministic error channel needs none of them.
* **Performance.** ``throw`` is not zero-overhead: even when an exception is
  never thrown, code with exception handling must compute exception ranges,
  and throwing itself must walk a live-call stack. With herbceptions the
  success path is just a call plus a branch on a flag.
* **Binary bloat.** Landing pads, type tables, and unwind tables add
  significant object-code size.
* **Not freestanding-friendly.** Exception handling depends on runtime ABI
  support (``<cxxabi.h>``, personality routines, ``__cxa_*``). The
  deterministic channel is a plain calling-convention feature and works in
  freestanding environments.

Keyword naming
--------------

The original C standard proposal (N2289) used ``return_failure{E}`` and ``failure(expr)``
as the keyword names for the C-style error specifier and error-return
expression. These names were rejected in this implementation for two reasons:

* ``failure`` collides with ``std::failure``, the standard exception class
  defined in ``<ios>``/``<system_error>``. Making ``failure`` a keyword breaks
  any code that includes ``<iostream>``, ``<system_error>``, or any header that
  transitively defines ``std::failure`` — a non-starter for a feature meant to
  interoperate with existing C++ code.
* Both ``return_failure`` and ``failure`` are common English words that appear as
  identifiers in real code (variable names, function names, member names).
  Keywords must be distinctive; reserving common words imposes an unacceptable
  migration burden on existing codebases.

This implementation uses ``return_failure`` instead. It serves both roles:

* ``return_failure{E}`` — function specifier (replaces ``return_failure{E}``).
* ``return_failure expr;`` — statement (replaces ``failure(expr)``). Parsed as
  a standalone statement like ``return`` or ``throw``; the operand is an
  expression and no parentheses are required. Also valid:
  ``return_failure(expr);``.

The keyword reads naturally in context: ``return_failure x;`` mirrors
``return x;`` and ``throw x;``. It is distinctive, does not collide with any
standard library identifier, and is unlikely to appear in existing code.

Language-level design
=====================

Function specifiers
-------------------

Two specifiers exist and cannot coexist on the same function:

.. code-block:: cpp

   // C++ only. Implicit error type std::error.
   T foo() throws;

   // C++ and C. Explicit error type E (trivially copyable).
   T foo() return_failure{E};

Semantically:

.. code-block:: cpp

   T foo() throws;                    // may fail via herbceptions channel; nounwind
   T foo() return_failure{E};         // may fail via herbceptions channel; nounwind

* ``throws`` and ``return_failure{E}`` are part of the *canonical function type* and
  change the function ABI (the return type is lowered to ``{T, i1}``). This
  is unlike ``noexcept`` since C++17, which is not part of the canonical
  type.
* ``throws(false)`` means the function cannot fail: it is equivalent to
  ``noexcept(true)``.
* ``throws``/``return_failure{E}`` and ``noexcept`` are mutually exclusive: a
  function picks one or the other, never both. ``throws`` supersedes ``noexcept``.
* In trait queries (``noexcept(expr)``, ``is_nothrow``), ``throws(true)`` / bare
  ``throws`` / ``return_failure{E}`` behave as ``noexcept(false)`` because the
  function may fail via the herbceptions channel. However, they are not
  semantically ``noexcept(false)`` — they use a different ABI and a different
  exception mechanism. In LLVM IR they are ``nounwind`` (no unwinding).
* ``return_failure(E)`` (parentheses) is rejected; the braces form is mandatory.
  Combining ``throws`` and ``return_failure{...}`` on one declaration is rejected.

Restrictions
------------

* ``throws`` is only available in C++.
* ``return_failure{
  E}`` is a C-style feature restricted to **free functions**: member
  functions, lambdas, and coroutines cannot declare it.
* ``throws`` is **not allowed on coroutine declarations**; every herbception
  must be caught within the coroutine body.
* Destructors cannot be declared with a herbception specification.
* ``return_failure{
  std::error}`` is rejected: the implicit error type of ``throws``
  is compiler-fabricated and cannot be named explicitly.
* ``E`` must be trivially copyable.

Function pointers
`````````````````

Because the specifier changes the canonical type, function pointers with
``throws``/``return_failure`` are distinct types. There are **no implicit
conversions** in either direction to/from plain function pointers, and a
virtual override must use the same specifier as the base.

Error domains
`````````````

An error *type* must be registered with a ``std::error_domain``
specialization exposing at least a non-null ``domain()`` singleton pointer
and a ``code(E)`` projection. ``T`` need not be an enum -- any type with an
``error_domain<T>`` specialization can be thrown:

.. code-block:: cpp

    namespace std {
  enum class my_errc : unsigned { ok = 0, bad = 1 };
  template <> struct error_domain<my_errc> {
    static constexpr error_domain_singleton const *domain() noexcept;
    static constexpr unsigned long code(my_errc e) noexcept {
      return static_cast<unsigned long>(e);
    }
  };
    }

Returning ``nullptr`` from ``domain()`` is a compile-time error: the
fabricated ``std::error`` dereferences the domain pointer in its destructor.

The ``libherbceptions`` runtime ships ready-made domains -- POSIX (``errc``),
Win32, NTSTATUS, COM, Wine, ``cmath`` and ``parse`` -- each exposed through
an extern ``"C"`` factory (``__cxa_error_domain_posix()``,
``__cxa_error_domain_nt()``, ...) declared in the ``std::error_domains``
namespace of ``<herbceptions/error>``, plus the exception-pointer domain
used for legacy-EH interop (see `Traditional exceptions`_).

Returning an error
------------------

A ``throws`` function returns an error with ``throw throws expr`` (C++
only); a ``return_failure{
  E}`` function returns an error with ``return_failure(expr)``
instead -- ``throw throws`` is not available inside ``return_failure{
  ...}``
functions:

.. code-block:: cpp

   int read_file(const char *path) throws {
  FILE *f = fopen(path, "rb");
  if (!f)
    throw throws std::errc(errno);
  return fileno(f);
   }

The operand is the error value; the compiler fabricates the (otherwise
unconstructible) ``std::error`` by evaluating
``std::error_domain<T>::domain()`` and ``std::error_domain<T>::code(e)``.
An operand whose type has no ``std::error_domain`` specialization is
rejected. Inside a ``catch throws`` handler the operand form is allowed
only when the enclosing function itself declares ``throws`` /
``return_failure{
  ...}`` (the new error then leaves via its own channel); otherwise
use the bare rethrow instead (see below).

A ``return_failure{
  E}`` function returns an error with ``return_failure(expr)``
(C and C++), where ``expr`` has exactly the type ``E``:

.. code-block:: c

   int divide(int a, int b) return_failure{int} {
   if (b == 0)
     return_failure 42;
  return a / b;
   }

Bare ``throw throws`` (without an operand) rethrows the error currently
being handled and is only valid inside a ``catch throws`` handler.

Calling a function that can fail
--------------------------------

**C++** -- a bare call to a ``throws``/``return_failure`` function inside a function
that itself has a ``throws``/``return_failure`` specifier **auto-propagates** the
error; no wrapper is required:

.. code-block:: cpp

   int process(int x) throws {
  int fd = open_wrapped(x); // auto-propagates on failure
  return fd;
   }

The auto-propagation is suppressed while parsing the operand of an explicit
``try(expr)`` or ``catch return_failure(expr)``.

**C** -- calling a ``return_failure{
  E}`` function without an explicit wrapper is a
compile-time error. The error must be handled with ``try(expr)`` or
``catch return_failure(expr)``:

.. code-block:: c

   // error: calling function with 'return_failure{...}' specifier requires
   //        'try()' or 'catch return_failure()' wrapper
   int x = some_return_failure_func();

   int x = try(some_return_failure_func());                    // auto-propagate
   struct {
  union {
    int value;
    int error;
  };
  bool failed; }
     e = catch return_failure(some_return_failure_func());               // inspect

Explicit handling
-----------------

``try(expr)`` -- evaluates ``expr`` (a call to a ``throws``/``return_failure``
function); on failure it auto-propagates the error; on success it yields the
success value. It is only valid inside a function that itself declares
``throws``/``return_failure{
  ...}``:

.. code-block:: cpp

   int foo(int x) throws {
  return try
    (bar(x)) + 1; // if bar return_failure, foo return_failure with bar's error
   }

``catch return_failure(expr)`` -- evaluates ``expr`` and produces the N2289 aggregate
``struct {
  union {
    T value;
    E error;
  };
  bool failed; }``. ``value`` and
``error`` are accessible through the anonymous union; ``failed`` is false on
success and true on error. It cannot be applied to a plain ``throws``
function (whose implicit ``std::error`` can only be handled by a
``catch throws`` block handler):

.. code-block:: cpp

   auto e = catch return_failure(bar(x));
   if (e.failed) {
  handle(e.error);
   } else {
  use(e.value);
   }

Catching errors with a block
----------------------------

``catch throws(std::error e) {
  ... }`` provides block-based handlers. The
handler must declare exactly ``std::error``, by value: references,
cv-qualified forms, other types and ``catch throws(...)`` are rejected, and
there is no block form of ``catch return_failure`` (it exists only as an expression).
A bare call to a ``throws``/``return_failure`` function inside the ``try`` block
routes the error value to the handler instead of propagating:

.. code-block:: cpp

   try {
  try
    foo(); // foo() throws
   } catch throws(std::error e) {
  if (e == std::errc::no_such_file_or_directory) {
    ...
  }
   }

Inside the handler, bare ``throw throws`` rethrows the caught error.

Additional rules for ``catch throws`` handlers:

* Herbception and traditional catch clauses may be mixed and interleaved in
  one try statement; the two channels dispatch independently. Legacy C++
  exceptions match only the traditional clauses (in their relative order,
  with ``catch(...)`` last among them); herbception errors scan only the
  ``catch throws`` handlers in their relative order.
* A ``throw throws`` inside a *traditional* handler chains to the next
  herbception handler after it.
* When a try has no traditional clauses, a ``catch throws(std::error)``
  handler also receives legacy exceptions auto-converted through the
  exception-pointer domain; with traditional clauses present they are
  delivered untouched instead.
* Inside a ``return_failure{
  E}`` function, a ``catch throws(std::error)`` handler
  requires a visible ``std::error_domain<E>`` specialization.
* Inside such a handler, a call to a plain ``return_failure{
  ...}`` function must be
  wrapped in an explicit ``try()`` so its error is converted to
  ``std::error`` (C-style explicitness):

.. code-block:: cpp

   void g() return_failure{std::errc} {
  try {
    // ...
  } catch throws(std::error e) {
    auto r = try
      (return_failure_callee()); // ok: converted via error_domain
    // return_failure_callee();              // rejected: unconverted raw payload
  }
   }

Convertibility between specifiers
`````````````````````````````````

* Calling a ``return_failure{
  E}`` function from a ``throws`` function (by bare call
  or ``try(expr)``) converts the error to ``std::error`` through the
  ``std::error_domain<E>`` accessors (``domain()`` / ``code()``); a missing
  specialization is rejected.
* Calling a ``throws`` function from a ``return_failure{
  E}`` function propagates the
  fabricated two-word ``std::error`` payload verbatim into the error slot;
  no conversion is performed.

``noexcept`` boundary
---------------------

A herbception error must never silently escape a ``noexcept(true)``
function. Calling a ``throws`` function without handling it from a function
that is neither ``throws`` nor ``return_failure{
  ...}`` is a diagnostic, and
``try foo()`` (which propagates) is likewise rejected there; use a
``try { } catch throws(...)`` block instead. ``int main()`` is a special
case: an unhandled error terminates via ``llvm.trap``.

Traditional exceptions
----------------------

A ``throws`` function implicitly **converts any legacy C++ exception that
escapes it** (thrown by a ``noexcept(false)`` callee) into a fabricated
``std::error`` on the herbception channel, exactly like a
``catch throws(std::error)`` handler does inside a ``try`` block. The
fabricated value pairs the exception-pointer domain singleton with the
caught-object pointer as its code, via the ``libherbceptions`` ABI entry
points ``__cxa_error_domain_{itanium,msvc}_exception_ptr()`` /
``__cxa_error_code_{itanium,msvc}_exception_ptr(ptr)``. Destructors still
run normally (during unwinding, since ``throws`` calls are plain
calls/invokes).

The conversion uses the built-in ``libherbceptions`` ABI entry points
``__cxa_error_domain_{itanium,msvc}_exception_ptr()`` /
``__cxa_error_code_{itanium,msvc}_exception_ptr(ptr)``, baked into the
compiler. The linker hard-errors if ``libherbceptions`` is not linked.
Destructors still run normally (during unwinding, since ``throws`` calls are
plain calls/invokes).
exceptions entirely.

Templates and concepts
----------------------

``try(expr)`` and ``catch return_failure(expr)`` accept dependent calls. Inside a
template the check is deferred to instantiation, and the herbception flag on
``throw throws`` is preserved when the expression is rebuilt, so
instantiated templates behave correctly. Constrained templates (concepts)
work as usual.

Constexpr
---------

``return_failure{
  E}`` functions, ``throw throws``, ``try(expr)`` auto-propagation,
``catch return_failure(expr)`` and ``try { } catch throws(std::error)`` blocks are
usable in constant expressions:

.. code-block:: cpp

   constexpr int f(int x) return_failure{int} {
  if (x == 0)
    return_failure(42);
  return 2 * x;
   }
   constexpr int g(int x) return_failure{int} {
  return try
    (f(x)); }
   static_assert(g(3) == 6);
   static_assert(g(0) == 42);

At compile time the fabricated domain pointer is a unique opaque constant,
so ``e.code()`` and ``e == errc_value`` comparisons work in constant
expressions.

Coroutines
----------

Coroutines cannot declare ``throws`` or ``return_failure{
  ...}``; all herbceptions
must be caught within the coroutine body.

Feature-test macro
------------------

``-fherbceptions`` defines ``__HERBCEPTIONS__`` so code can detect the
feature:

.. code-block:: cpp

#ifdef __HERBCEPTIONS__
   int foo() throws;
#endif

Type traits
-----------

``-fherbceptions`` provides compiler builtins and the corresponding
``std`` traits (in ``<herbceptions/error>``) for querying the type system:

.. code-block:: cpp

   // T can be thrown via `throw throws`: a usable error_domain<T> exists.
   static_assert(__is_herbceptions_throwsable(std::my_errc));

   // Function type is declared `return_failure{E}` (not plain `throws`).
   static_assert(__is_invoke_herbceptions_return_failure(decltype(f)));

   // { value_type, error_type } of an invoke-return_failure function type.
   using R = __invoke_herbceptions_return_failure_result<decltype(f)>;

Additional traits mirror the classic ``traits`` family along the herbception
channel: ``__is_herbceptions_throws_constructible``,
``__is_herbceptions_throws_assignable``,
``__is_herbceptions_throws_convertible``,
``__has_herbceptions_throws_constructor``, ``__has_herbceptions_throws_copy``,
``__has_herbceptions_throws_assign`` and
``__has_herbceptions_throws_move_assign``. The type trait
``__invoke_herbceptions_return_failure_t`` yields the raw ``{
  T, i1}``-shaped result
type of a ``return_failure`` function type.

Convenience specializations of ``__is_herbceptions_throws_constructible`` mirror
the standard ``is_nothrow_copy_constructible`` / ``is_nothrow_move_constructible``
family::

   // Copy/move construction that can propagate a herbception.
   static_assert(std::is_herbceptions_throws_copy_constructible_v<T>);
   static_assert(std::is_herbceptions_throws_move_constructible_v<T>);

Invocation traits test whether a callable can propagate a herbception error
through the ``throws`` channel when invoked::

   // F is invocable with Args... and the call can throw.
   static_assert(std::is_herbceptions_throws_invocable_v<F, Args...>);
   // Same, and the result is convertible to R.
   static_assert(std::is_herbceptions_throws_invocable_r_v<R, F, Args...>);

Runtime support
===============

The ``libherbceptions`` runtime provides the ``error_domain_singleton``
vtables for the standard domains and the ``std::error`` class. Its API
(``<herbceptions/error>``):

.. code-block:: cpp

   class error {
  // Only the compiler fabricates std::error values: default/copy/move
  // construction and assignment are deleted, and ~error() runs the
  // domain's do_cleanup. The private payload is exactly two words
  // ({const error_domain_singleton*, size_t}) so it flows through the
  // {void*, size_t} ABI slot unchanged.
  [[nodiscard]] constexpr error_domain_singleton const *domain() const noexcept;
  [[nodiscard]] constexpr std::size_t code() const noexcept;
  template <class T> constexpr bool equivalent(T ec) const noexcept;
  constexpr std::errc to_errc() const noexcept;
  void throw_dynamic_exception() const; // rethrow as a traditional exception
  template <class T> constexpr bool is_code_of() const noexcept;
   };

   struct error_domain_singleton {
  void (*do_cleanup)(std::size_t) noexcept;
  bool (*do_equivalent)(std::size_t, error_domain_singleton const *,
                        std::size_t) noexcept;
  void (*do_query_information)(std::size_t, error_query_information,
                               error_reporter_encoding, void *,
                               error_reporter_io_cookie_function) noexcept;
  std::errc (*do_to_errc)(std::size_t) noexcept;
  void (*do_throw_dynamic_exception)(std::size_t, std::dynamic_exception_abi);
   };

``do_query_information`` produces a domain *name* and/or *message* for an
error code, as a writev-style list of scatter pieces, encoded on request as
UTF-8/UTF-16/UTF-32 (e.g. ``[posix]Owner died``), so any printer can render
errors without hard-coding domain knowledge.

LLVM IR representation
======================

A function declared ``throws`` / ``return_failure{
  E}`` is lowered with the LLVM
``throws`` attribute and a struct return:

.. code-block:: llvm

   ; C++: T foo() throws;
   define {
  T, i1 } @foo(...) #0 { ... }

   ; C: void foo() return_failure{E};
   define {
  E, i1 } @foo(...) #0 { ... }

   attributes #0 = { throws }

The ``i1`` discriminant is ``false`` for success and ``true`` for error. The
payload slot is a value-or-error union of size ``max(T, E)``: on success it
holds ``T``, on failure the error value. For the implicit ``std::error`` type
(a 2-register ``{
  void *, size_t}`` struct) a ``T throws`` function returns
``{
  {ptr, i64}, i1 }``.

Call-site lowering
------------------

Callers of a ``throws`` function:

1. emit the call;
2. check the discriminant;
3. on success use the payload as the value;
4. on failure either propagate (return the error with the discriminant set)
   or branch to a handler (``catch return_failure`` / ``catch throws``).

Because a ``throws`` call is an ordinary call, there is no ``invoke``, no
landing pad, and no personality function on the pure-herbception path.
Cleanups run on the normal scope-exit path.

Legacy-EH interop (Itanium / MSVC / Wasm / SjLj)
------------------------------------------------

A ``throws`` function that can call ``noexcept(false)`` functions is wrapped
in a whole-function catch-all EH scope. Calls to ``noexcept(false)``
functions inside it become ``invoke``\ s into a landing pad whose handler
fabricates the ``std::error`` (via the ``libherbceptions`` exception-ptr ABI
symbols and ``__cxa_get_exception_ptr`` / ``llvm.eh.exceptionpointer`` /
``wasm.get.exception``) and routes it to the throws return path. The same
conversion is applied inside a ``try { } catch throws(std::error)`` block.

A default ``return_failure{E}`` function pushes a terminate landing pad
(``noexcept``-like semantics for legacy C++ exceptions): any legacy
exception escaping it calls ``std::terminate``.

Target-specific discriminant
============================

When the target reports ``supportThrowsCC()``, the backend can carry the
discriminant in a target-specific location instead of an extra register in
the struct return:

.. list-table::
   :header-rows: 1

   * - Target
     - Discriminant
     - Callee sets success / error
     - Caller checks
   * - x86-64
     - Carry flag (``CF`` in EFLAGS)
     - ``clc`` / ``stc``
     - ``setb`` / ``jc``
   * - AArch64
     - Carry flag (``C`` in NZCV)
     - ``subs xzr, xzr, xzr`` / ``subs xzr, xzr, #1``
     - ``cset w0, hs``
   * - ARM (32-bit)
     - Carry flag (``C`` in CPSR)
     - carry-flag set/clear
     - ``cset`` / conditional branch
   * - RISC-V
     - Extra integer register (``a2``)
     - ``li a2, 0`` / ``li a2, 1``
     - ``beqz a2, success``
   * - LoongArch
     - Extra integer register (``$a2``)
     - ``li $a2, 0`` / ``li $a2, 1``
     - ``beqz $a2, success``
   * - WebAssembly
     - Extra multivalue result
     - second result ``0`` / ``1``
     - branch on the extra result

On targets without ``supportThrowsCC()``, the discriminant falls back to the
``{
  T, i1}`` struct return (two registers or sret).

Implementation notes
====================

The implementation spans the following areas:

* **Parsing** -- ``throws`` / ``return_failure{
  E}`` (including the delayed-parsing
  path and the ``noexcept`` combination checks) in
  ``clang/lib/Parse/ParseDeclCXX.cpp``; ``try(expr)``,
  ``catch return_failure(expr)`` dispatched in ``clang/lib/Parse/ParseExpr.cpp`` and
  built in ``ParseExprCXX.cpp`` alongside ``throw throws``;
  ``catch throws(E e)`` block handlers in
  ``clang/lib/Parse/ParseStmt.cpp``. ``try``, ``catch``, ``throws``,
  ``return_failure`` and ``failure`` carry the ``KEYHERB`` keyword flag so they parse
  in C with ``-fherbceptions``.
* **Sema** -- ``ActOnHerbceptionTry``, ``ActOnHerbceptionCatchFails``,
  ``ActOnCXXThrowThrows``, ``ActOnHerbceptionFailure``,
  ``BuildCxaExceptionErrorValue`` in ``clang/lib/Sema/SemaExprCXX.cpp``;
  ``ActOnCXXCatchThrowsBlock`` in ``SemaStmt.cpp``; auto-propagation of bare
  calls wired in ``ActOnCallExpr`` (``SemaExpr.cpp``), suppressed while
  parsing the operand of an explicit wrapper via
  ``Sema::HerbceptionOperandDepth``; C rejects bare calls with
  ``err_return_failure_call_without_wrapper``.
* **AST** -- ``CXXTryExpr``, ``CXXCatchFailsExpr``, ``CXXCatchThrowsStmt``,
  ``CXXErrorValueExpr`` and ``CXXCxaExceptionExpr`` nodes; the
  ``EST_BasicThrows`` / ``EST_ThrowsTyped`` /
  ``EST_ThrowsTypedNoexceptFalse`` exception-specification kinds; the N2289
  ``catch return_failure`` aggregate built by ``ASTContext::getCatchFailsType``.
* **CodeGen** -- ``{
  T, i1}`` return lowering with a payload sized
  ``max(T, E)``, the ``throws`` attribute, the whole-function legacy-EH
  conversion scope (``EmitStartEHSpec`` / ``getHerbceptionLegacyConvert`` in
  ``clang/lib/CodeGen/CGException.cpp``), ``catch throws``/``catch return_failure``
  routing, the ``main()`` trap, and the carry-flag / extra-register
  discriminants in the backends.
* **Runtime** -- ``libherbceptions``: the ``error_domain_singleton`` vtables
  (posix, win32, nt, com, wine, cmath, parse, exception-ptr), ``std::error``,
  and the name/message query protocol.

Link-time ODR checking
======================

The ``throws``/``return_failure`` specifier is deliberately not part of the mangled
name: a herbception function and its plain counterpart share one symbol.
That keeps object files link-compatible with non-herbception toolchains,
but it also means translation units that *disagree* about a function's
specifier (or about a ``return_failure{
  E}`` error type) compile cleanly and then
link silently -- callers compiled against the plain ABI would read garbage
from a ``throws`` definition. That is a One Definition Rule violation, and
only the linker can see it.

When you link with LTO (``-flto``, full or thin), ``ld.lld`` compares the
herbception signature -- whether the IR function carries the error channel
and which payload type it returns -- of every externally visible function
across all bitcode modules, for every supported target, and return_failure the link
on a conflict:

.. code-block:: none

   ld.lld: error: herbception ODR violation: symbol '_Z3fooi' has
   conflicting definitions: it is defined as a herbception ('throws')
   function with error payload type '{ ptr, i64 }' in 'a.o', but without
   the herbception error channel in 'b.o'

The check covers both full LTO and ThinLTO. It only sees bitcode inputs:
definitions coming from native relocatable files (or from modules never
materialized in-process, e.g. with ``--thinlto-index-only`` or served from
the ThinLTO cache) are outside its reach.

Known limitations
=================

* The ABI is experimental and not stable across compiler versions.
* ``FastISel`` falls back to SelectionDAG for ``throws`` calls, so some
  ``-O0`` paths are slightly slower than they would otherwise be.
* Legacy C++ exception conversion depends on the ``libherbceptions``
  runtime ABI symbols baked into the compiler; the linker hard-errors if
  ``libherbceptions`` is not linked.

Related work
============

* Herb Sutter, `P0709R4: Zero-overhead deterministic exceptions
  <https://wg21.link/p0709r4>`_ -- the design this extension follows.
* Swift error handling and the LLVM ``swifterror`` attribute.
* Rust ``Result<T, E>`` and Go multi-value error returns.

.. _herbceptions-implementation:

Herbceptions Implementation Notes
=================================

.. contents::
   :local:

This document describes the actual implementation of the herbception
``throws`` / ``return_failure{E}`` extension in Clang/LLVM and the accompanying
``libherbceptions`` runtime. It is written for Clang/LLVM developers; for
the user-facing language description and motivation, see
:ref:`herbceptions`.

Overview
========

Herbceptions are a deterministic error channel layered on the normal return
path. A function declared ``throws`` (implicit ``std::error``) or
``return_failure{E}`` (explicit error type) is lowered so that:

* the IR return type becomes ``{ T, i1 }`` (payload + discriminant);
* the LLVM function carries the ``throws`` attribute;
* on success the payload holds ``T`` and the discriminant is ``false``;
* on failure the payload holds the error value and the discriminant is
  ``true``.

There is no unwinder, landing pad, or personality function on the
pure-herbception path. Traditional C++ exceptions are supported only
through explicit interop points (see `Legacy C++ EH interop`_).

Coroutines cannot be declared ``throws`` / ``return_failure{...}``: every herbception
must be caught within the coroutine body
(``err_throws_not_allowed_in_coroutine``, diagnosed in
``Sema::ActOnCoroutineBodyStart``, ``clang/lib/Sema/SemaCoroutine.cpp``).

Language surface
================

Keywords and parsing
--------------------

The extension is gated behind ``LANGOPT(HerbExceptions)``
(``clang/include/clang/LangOptions.def``), driven by the driver flag
``-fherbceptions`` / ``-fno-herbceptions``
(``clang/include/clang/Options/Options.td``, forwarded in
``clang/lib/Driver/ToolChains/Clang.cpp``; independent from
``-fexceptions``). The flag also predefines the ``__HERBCEPTIONS__`` macro
(``clang/lib/Frontend/InitPreprocessor.cpp``).

The keywords carry the ``KEYHERB`` token key
(``def KEYHERB : TokenKey<0x80000000>`` in
``clang/include/clang/Basic/BuiltinTraits.td``): ``throws``, ``fails`` and
``failure`` are plain ``KEYHERB``; ``try`` and ``catch`` are
``KEYCXX|KEYHERB`` so they also parse in C with ``-fherbceptions``
(``TokenKinds.def``).

* Specifiers: ``Parser::tryParseExceptionSpecification``
  (``ParseDeclCXX.cpp``) accepts ``throws`` (C++ only, else
  ``err_throws_requires_cxx``) producing ``EST_BasicThrows``, evaluates
  ``throws(expr)`` like ``noexcept(expr)`` via ``Sema::ActOnThrowsSpec``
  (``throws(false)`` is ``EST_BasicThrowsFalse``), parses ``return_failure{E}``
  by storing ``E`` in the exception-type slot (``return_failure(E)`` is rejected with
  ``err_return_failure_paren_not_allowed``). ``return_failure{E}`` and ``noexcept``
  cannot be combined: throws supersedes noexcept. A delayed-parsing path handles
  the same forms after a trailing return type.
  ``tryParseNoexceptAfterThrows`` / ``tryParseNoexceptAfterFails`` reject
  ``throws`` + ``noexcept(false)`` (``err_throws_noexcept_false``) and the
  ``throws`` + ``fails`` combination (``err_throws_fails_combined``).
* Expressions: ``Parser::ParseAssignmentExpression`` dispatches
  ``try(expr)`` -> ``ParseHerbceptionTryExpression``, ``catch return_failure(expr)``
  -> ``ParseHerbceptionCatchFailsExpression`` and ``return_failure(expr)`` ->
  ``ParseHerbceptionFailureExpression`` (``ParseExpr.cpp``); all three are
  implemented in ``ParseExprCXX.cpp``. The try/catch-return_failure parsers bracket
  their operand with ``Actions.HerbceptionOperandDepth`` so auto-propagation
  is suppressed inside an explicit wrapper. ``throw throws`` is handled
  inside ``Parser::ParseThrowExpression``.
* Block handlers: ``Parser::ParseCXXCatchBlock`` recognizes
  ``catch throws(std::error e)`` and calls
  ``Actions.ActOnExceptionDeclarator(..., /*IsHerbception=*/true)`` followed
  by ``Actions.ActOnCXXCatchThrowsBlock``. A ``catch return_failure(...)`` token
  sequence is rejected at parse time
  (``err_catch_return_failure_expression_only``): ``catch return_failure`` exists
  only as an expression.

Function specifiers
-------------------

``throws`` is a C++-only specifier with the implicit error type
``std::error``. ``return_failure{E}`` works in C and C++ and carries the explicit
error type ``E``. Both are stored in the function type's exception
specification:

* ``EST_BasicThrows`` -- bare ``throws`` (and ``throws(true)``);
  ``FunctionProtoType::hasBasicThrowsSpec()``.
* ``EST_ThrowsTyped`` -- ``return_failure{E}``; ``E`` is stored in the exception-type
  slot (``hasReturnFailureSpec()``).
* ``throws(false)`` is ``EST_BasicThrowsFalse``: cannot fail via herbceptions,
  equivalent to ``noexcept(true)``.

``throws``/``return_failure{E}`` and ``noexcept`` are mutually exclusive;
the parser rejects any combination. Semantic checks live in
``Sema::checkExceptionSpecification`` /
``actOnDelayedExceptionSpecification`` (``SemaDeclCXX.cpp``):
``return_failure{std::error}`` is rejected (``err_return_failure_std_error_type``), ``E`` must
be trivially copyable (``err_return_failure_type_not_trivially_copyable``),
destructors cannot carry a herbceptions spec
(``err_herbceptions_destructor_spec``) and ``return_failure{E}`` is restricted to free
functions (``err_return_failure_only_free_function`` -- enforced for members via the
delayed-spec path and for coroutines in ``ActOnCoroutineBodyStart``).

The specifiers are part of the canonical function type because they change
the calling convention (``{T, i1}`` return). Function pointers, overload
resolution and virtual overrides therefore treat them as distinct
(``err_herbceptions_spec_mismatch`` /
``err_herbceptions_override_spec_mismatch``). ``FunctionProtoType::canThrow()``
returns a dedicated ``CT_Deterministic`` for ``EST_BasicThrows`` /
``EST_ThrowsTyped``. The type printer renders the specifiers as
``" throws"`` / ``" return_failure{E}"`` (``TypePrinter.cpp``).

Expressions and statements
--------------------------

* ``throw throws expr`` -- ``Sema::ActOnCXXThrowThrows``
  (``clang/lib/Sema/SemaExprCXX.cpp``). Only valid inside a function with a
  plain ``throws`` spec (a ``return_failure{E}`` function must use
  ``return_failure(...)`` instead; ``err_throw_throws_in_fails_function``)
  or inside a ``try`` block / catch clause
  (``err_throw_throws_outside_throws_function``). With an explicit operand
  it creates a *new* error: unless the enclosing function is ``return_failure{E}``,
  the compiler fabricates the unconstructible ``std::error`` through
  ``error_domain<T>::domain()`` / ``code(e)`` (missing specialization ->
  ``err_throw_throws_no_error_domain``). Inside a catch-throws handler body
  the herbception catch scopes are already deactivated (CodeGen pops them
  before emitting handlers), so the operand form additionally requires the
  enclosing function to have a throws/return_failure spec
  (``err_throw_throws_no_catch_handler``); bare ``throw throws`` rethrows
  from the handled error slot and is valid nowhere else.
* bare ``throw throws`` -- rethrow; only valid inside a ``try`` block whose
  handlers are herbception handlers
  (``err_throw_throws_rethrow_outside_catch``). CodeGen reads the error from
  the active catch scope's error slot.
* ``try(expr)`` -- ``Sema::ActOnHerbceptionTry`` builds ``CXXTryExpr``. Only
  valid inside a throws/return_failure function
  (``err_try_throws_outside_throws_function``); the operand must be a call
  to a throws/return_failure function (``err_try_expr_requires_throws_call``,
  deferred while type-dependent). When a ``throws`` caller invokes a
  ``return_failure{E}`` callee, the resolved ``error_domain<E>`` record is attached
  to the node for the E->std::error conversion on the error path.
* ``catch return_failure(expr)`` -- ``Sema::ActOnHerbceptionCatchReturnFailure`` builds
  ``CXXCatchReturnFailureExpr`` holding the N2289 aggregate type produced by
  ``ASTContext::getCatchReturnType(T, E)``:
  ``struct { union { T value; E error; }; bool failed; }`` (an implicit
  record named ``__herb_catch_return_failure``). A plain ``throws`` callee is
  rejected (``err_catch_return_failure_expr_throws_function``).
* ``return_failure(expr)`` -- ``Sema::ActOnHerbceptionReturnFailure``; only valid inside
  a ``return_failure{E}`` function (``err_failure_outside_return_failure_function``) with an
  operand of type ``E``; lowers to the same path as ``throw throws``
  (``BuildCXXThrow(..., /*IsHerbception=*/true)``).
* ``try { ... } catch throws(std::error e) { }`` -- checked by
  ``Sema::BuildExceptionDeclaration`` / ``ActOnExceptionDeclarator``
  (``IsHerbception``) and ``Sema::ActOnCXXCatchThrowsBlock``
  (``SemaStmt.cpp``), which builds ``CXXCatchThrowsStmt``. The handler must
  bind exactly ``std::error``, by value: references, cv-qualifiers, other
  types and the ellipsis form are rejected
  (``err_catch_throws_std_error`` / ``err_catch_throws_ellipsis``), and a
  block-form ``catch return_failure`` is rejected at parse time
  (``err_catch_return_failure_expression_only``). ``Sema::ActOnCXXTryBlock`` detects
  try blocks with herbception handlers and skips them from EH type-matching
  and the "exception used but not catchable" diagnosis.

Auto-propagation
----------------

In ``Sema::ActOnCallExpr`` (``clang/lib/Sema/SemaExpr.cpp``): in C++, when
the current function has a throws/return_failure spec and
``Sema::HerbceptionOperandDepth == 0``, a bare call to a throws/return_failure
function is wrapped in ``ActOnHerbceptionTry`` (auto-propagation). In C,
any unwrapped call is rejected with ``err_return_failure_call_without_wrapper`` plus
``note_return_failure_function_declared_here``. Inside a ``catch throws(std::error)``
handler of a ``return_failure{E}`` function, a bare call to a plain ``return_failure{E2}``
function is rejected outright (``err_return_failure_call_in_catch_throws``): the
handler slot holds std::error, so the raw E2 payload would be stored
unconverted; an explicit ``try()`` performs the conversion.

``noexcept`` boundary
---------------------

A call whose result would escape a non-throws/non-return_failure enclosing function
is diagnosed at call-lowering time with
``err_herbceptions_non_throws_call_throws`` (``CodeGen::EmitCall``,
``clang/lib/CodeGen/CGCall.cpp``). ``main()`` is a special case: its error
path branches to a ``herb.main.trap`` block that executes ``llvm.trap``
(the success path continues in ``herb.main.ok``).

Type traits
-----------

Defined in ``clang/include/clang/Basic/BuiltinTraits.td`` and implemented in
``clang/lib/Sema/SemaTypeTraits.cpp``:

* ``__is_herbceptions_throwsable(T)`` -- ``T`` has a usable
  ``error_domain<T>``.
* ``__is_invoke_herbceptions_fails(F)`` -- ``F`` is a ``return_failure{E}`` function
  type.
* ``__invoke_herbceptions_fails_result<F>`` (builtin template,
  ``BuiltinTemplates.td``; cached result type via
  ``ASTContext::getInvokeHerbceptionsFailsResultType``) -- the
  ``{ value_type, error_type }`` pair.
* ``__invoke_herbceptions_fails_t<F>`` -- the raw ``{T, i1}``-shaped type
  trait.
* Herbception analogues of the classic traits:
  ``__is_herbceptions_throws_constructible/_assignable/_convertible`` and
  ``__has_herbceptions_throws_constructor/_copy/_assign/_move_assign``.

``libherbceptions/include/herbceptions/error`` mirrors these as
``std``-style trait aliases gated on ``__HERBCEPTIONS__``.

AST
===

Nodes
-----

All registered in ``clang/include/clang/Basic/StmtNodes.td``:

* ``CXXTryExpr`` (``ExprCXX.h``) -- herbception ``try(expr)``; distinct from
  the statement-level ``CXXTryStmt``. Carries the optional
  ``CXXRecordDecl *ErrorDomain`` used for the fails-to-std::error
  conversion.
* ``CXXCatchReturnFailureExpr`` (``ExprCXX.h``) -- ``catch return_failure(expr)``; wraps the
  call and the N2289 aggregate type.
* ``CXXCatchThrowsStmt`` (``StmtCXX.h``) -- a ``catch throws(E e)`` /
  ``catch return_failure(E e)`` handler. Stores the specifier location and the
  optional *legacy conversion expression*
  (``getLegacyExceptionErrorValue()``): the fabricated ``std::error`` for a
  caught traditional C++ exception, built by
  ``Sema::ActOnCXXCatchThrowsBlock`` only when the handler binds
  ``std::error`` and the conversion inputs are visible.
* ``CXXErrorValueExpr`` (``ExprCXX.h``) -- a compiler-fabricated
  ``{domain, code}`` value (domain/code accessor calls as operands) that
  users cannot construct.
* ``CXXCxaExceptionExpr`` (``ExprCXX.h``) -- the thrown-object pointer of a
  legacy exception being converted ("magic" expression lowered per
  personality).

Exception specification storage
-------------------------------

``FunctionProtoType`` stores the herbception specifier in
``FunctionTypeBits.ExceptionSpecType`` and, for ``return_failure{E}``, the error type
in the exceptions slot. Helpers on ``FunctionProtoType``: ``hasThrowsSpec()``,
``hasBasicThrowsSpec()``, ``hasFailsSpec()`` and the enum's
``hasHerbceptionExceptionSpec()`` (``clang/include/clang/AST/TypeBase.h``,
``clang/include/clang/Basic/ExceptionSpecificationType.h``). Because the
specifier changes the lowered return type it participates in the canonical
function type. ``FunctionDecl`` additionally carries the whole-function
legacy-conversion expression (see below).

Sema
====

``SemaExprCXX.cpp``
-------------------

* ``isHerbceptionThrowsCall`` -- whether an expression is a call to a
  throws/return_failure function (used by auto-propagation and the wrapper checks).
* ``lookupErrorDomain`` -- resolves ``std::error_domain<T>`` to a defined,
  user-provided specialization (implicit instantiations of the primary
  template are ignored so that missing specializations stay silent until
  they matter).
* ``findOrCreateImplicitExternCFunction`` -- declares-or-reuses an implicit
  extern ``"C"`` function; reused only when an existing declaration matches
  exactly.
* ``BuildCxaExceptionErrorValue`` -- builds the fabrication of the
  ``std::error`` capturing a legacy exception: calls to
  ``__cxa_error_domain_{itanium,msvc}_exception_ptr()`` (domain singleton)
  and ``__cxa_error_code_{itanium,msvc}_exception_ptr(ptr)`` (code minted
  from a ``CXXCxaExceptionExpr`` operand; MSVC variant takes no argument),
  producing a ``CXXErrorValueExpr`` typed as ``std::error``. Requires
  ``std::exception_ptr`` and ``std::error`` to be visible; otherwise returns
  ``ExprError`` silently (the catch-throws handler still catches herbception
  throws -- only the legacy-EH conversion is unavailable).
* ``ActOnHerbceptionTry``, ``ActOnHerbceptionCatchFails``,
  ``ActOnCXXThrowThrows``, ``ActOnHerbceptionFailure`` -- see
  `Expressions and statements`_. Template instantiation rebuilds these nodes
  via ``TreeTransform.h`` (``RebuildCXXTryExpr``,
  ``RebuildCXXCatchFailsExpr``, ``RebuildCXXErrorValueExpr``), preserving
  the herbception flags across instantiation.

``SemaStmt.cpp``
----------------

* ``ActOnCXXCatchThrowsBlock`` -- builds ``CXXCatchThrowsStmt``; attaches
  ``BuildCxaExceptionErrorValue``'s expression when the handler binds
  ``std::error``; rejects handlers in a ``return_failure{E}`` function whose ``E``
  has no visible ``std::error_domain`` specialization
  (``err_catch_throws_requires_error_domain``).
* ``ActOnCXXTryBlock`` -- marks try blocks containing
  ``CXXCatchThrowsStmt`` handlers so CodeGen routes the discriminant instead
  of using EH type-matching; the [except.handle]p5 catch-all-position check
  considers only later *traditional* clauses, so traditional and herbception
  clauses may interleave freely.

Whole-function conversion
-------------------------

A bare ``throws`` function implicitly converts any legacy C++ exception that
escapes it. ``Sema::ActOnFinishFunctionBody`` (``SemaDecl.cpp``):

* diagnoses a ``domain()`` returning ``nullptr`` inside a record named
  ``error_domain`` (``err_herbceptions_domain_nullptr``; the fabricated
  ``std::error`` dereferences it in ``~error()``);
* if the function is ``EST_BasicThrows`` and can call ``noexcept(false)``
  callees, builds ``BuildCxaExceptionErrorValue`` and stores it on the
  ``FunctionDecl`` via ``setHerbceptionLegacyErrorValue()``. The conversion
  uses built-in ABI symbols baked into the compiler; the linker hard-errors
  if libherbceptions is not linked. The member is serialized by
  ``ASTWriterDecl.cpp`` / ``ASTReaderDecl.cpp``.

Constexpr
---------

Constant evaluation supports the full channel
(``clang/lib/AST/ExprConstant.cpp``): pending error state
(``EvalInfo::HerbceptionErrorPending`` / ``HerbceptionErrorValue`` with a
per-domain opaque singleton map), ``throw throws`` and ``failure`` marking
failure, propagation through bare calls and ``try(expr)``, and evaluation of
``try { } catch throws(std::error)`` blocks (comparisons against
``e.code()`` / domain work because ``domain()`` evaluates to a unique opaque
constant). The newer bytecode interpreter covers ``CXXCatchFailsExpr``
(``clang/lib/AST/ByteCode/Compiler.cpp``).

CodeGen
=======

``{T, i1}`` lowering
--------------------

``clang/lib/CodeGen/CGCall.cpp``: ``FunctionProtoType::hasThrowsSpec()`` is
threaded through ``CGFunctionInfo`` (``HasThrowsReturn``) together with the
herbception error IR type (``getHerbceptionErrorType``). The return ABI is a
``{ union{T, E}, i1 }`` struct where:

* ``union{T, E}`` is the first element, sized to
  ``max(sizeof(T), sizeof(E))`` (``std::error`` is a 2-register
  ``{void*, size_t}`` struct, 16 bytes);
* the trailing ``i1`` is a **carry-flag placeholder**, never a
  memory-resident value. ``HERB_SETCCr`` (X86 backend) reads CF into the
  ``i1`` after each call, so CF survives the call sequence as the active
  tag:
  - ``CF = 0`` -> union holds ``T`` (the first ``sizeof(T)`` bytes are
    ``T``; the rest is padding when ``sizeof(T) < sizeof(E)``);
  - ``CF = 1`` -> union holds ``E`` (the first ``sizeof(E)`` bytes are
    ``E``; the rest is padding when ``sizeof(E) < sizeof(T)``).

Under opaque pointers ``T&`` / ``T&&`` / ``T*`` all lower to the same
``ptr`` first element, so they share the same
``{ {ptr, i64}, i1 }`` calling convention regardless of the C++
reference kind. The ``throws`` attribute
(``llvm::Attribute::Throws``, defined in ``llvm/IR/Attributes.td``; bitcode
kind ``ATTR_KIND_THROWS``) is added to the function and call site, and the
function epilogue inserts the discriminant into the returned struct.

When ``sizeof(union{T, E}) > 16`` (e.g. ``T`` is a 32-byte struct), the
union must be passed by hidden sret pointer (``RDI`` on x86_64 SysV)
rather than by value. The ``i1`` discriminant remains in CF -- the
sret-pointer path is purely about the payload size.

Call-site routing
`````````````````

``EmitCall`` extracts the discriminant after the call. A genuine two-field
struct return is never misclassified: only a second struct element of type
``i1`` is treated as a throws discriminant. Depending on the context:

* an enclosing ``catch throws`` / ``catch return_failure`` scope
  (``CodeGenFunction::HerbceptionCatchScopes``): coerce the error value into
  the handler's slot (through memory where needed) and branch to the
  handler;
* a ``throws``/``fails`` function: store the error value into the return
  slot, set the discriminant, run cleanups, branch to the return block --
  this is ``EmitHerbceptionThrow`` (``CGStmt.cpp``), which first checks the
  nearest active catch scope (so bare ``throw throws`` rethrows route to the
  innermost handler) and coerces the payload between ``T`` / ``E`` /
  ``std::error`` representations;
* ``main()``: trap on error (blocks ``herb.main.ok`` / ``herb.main.trap``);
* otherwise: diagnose with ``err_herbceptions_non_throws_call_throws``.

Every ``-fherbceptions`` function gets a ``herbception.disc`` alloca in
``StartFunction`` tagged with ``!coro.outside.frame`` metadata so the
discriminant never lives in a coroutine frame.

``try(expr)`` / ``catch return_failure(expr)``
-------------------------------------

``CodeGenFunction::EmitHerbceptionTry`` (``CGStmt.cpp``) emits the call into
``try.ok`` / ``try.err`` blocks, auto-propagating on the error path (running
cleanups). ``EmitHerbceptionCatchFails`` emits the N2289 aggregate: stores
``value`` and sets ``failed=false`` on success, stores ``error`` and sets
``failed=true`` on failure (anonymous-union-aware member lookup).
``EmitFailsErrorToStdError`` converts a ``return_failure{E}`` error to ``std::error``
on the error path by calling the resolved ``error_domain<E>::domain()`` /
``code()`` static members (honoring an optional ``domain_alias_type``).

Block handlers
``````````````

``clang/lib/CodeGen/CGException.cpp`` ``EmitCXXTryStmt`` routes try
statements containing ``CXXCatchThrowsStmt`` handlers to
``EmitHerbceptionCatchTry``. Traditional clauses (typed and ``catch(...)``)
may interleave with them: they are pushed as one regular ``EHCatchScope``
(legacy stream, relative order), while the herbception handlers each get a
handler block (``catch.throws``) and an error slot (``herb.error``) pushed
on ``HerbceptionCatchScopes`` (so bare calls inside the try body route to
them). Herbception errors scan the herbception handlers in order; legacy
exceptions match only the traditional clauses; the exception-ptr
auto-conversion catch-all is installed only when no traditional clause
competes for the legacy stream. While a traditional handler body runs, its
"next herbception handler" scope is active, so ``throw throws`` there chains
forward. Cleanups (including the caught variable's destructor, which runs
the domain's ``do_cleanup``) execute exactly once. Funclet-based
personalities keep the handler inside the proper funclet region.

Legacy C++ EH interop
=====================

``catch throws(std::error e)`` inside a try block
-------------------------------------------------

When a pure-herbception try (no traditional clauses) has a handler binding
``std::error`` that carries a legacy conversion expression,
``EmitHerbceptionCatchTry`` additionally pushes a catch-all EH
scope around the try block, so calls to ``noexcept(false)`` functions inside
become ``invoke``\ s into a landing pad. The handler block
(``herb.legacy.convert``) fabricates the ``std::error``
(``EmitErrorValueExpr`` + ``EmitCxaExceptionPtr`` in ``CGStmt.cpp``) and
routes it to the handler. Personality-dependent thrown-pointer extraction:

* Itanium / SjLj: ``__cxa_get_exception_ptr(exn.slot)``.
* MSVC: ``llvm.eh.exceptionpointer`` from the funclet ``catchpad`` (the
  handler body is rewritten into a ``catchret`` form).
* Wasm: ``wasm.get.exception`` (object already stored in ``exn.slot``).

Whole-function conversion
-------------------------

For a bare ``throws`` function with a stored conversion expression,
``EmitStartEHSpec`` (``CGException.cpp``) pushes a whole-function catch-all
EH scope whose handler is ``getHerbceptionLegacyConvert()``;
``emitHerbceptionLegacyConvertBody`` fabricates the ``std::error`` and calls
``EmitHerbceptionThrow``, routing it to the throws return path (a missing
conversion expression is a hard error there, mirroring the Sema check).
``FinishFunction`` emits the block if it was used. ``EmitStartEHSpec`` /
``EmitEndEHSpec`` handle the ``return_failure{E}`` terminate scope:
a default ``return_failure{E}`` (implies ``noexcept``-like semantics for legacy
exceptions) pushes a terminate landing pad.

Backend
=======

``TargetLowering::supportThrowsCC()`` (``llvm/include/llvm/CodeGen/
TargetLowering.h``) advertises target-specific discriminant carrying;
overridden true by X86, AArch64, ARM, RISC-V, LoongArch and WebAssembly.
``CallLoweringInfo::IsThrows`` threads the property through SelectionDAG
(``SelectionDAGBuilder.cpp``), FastISel and GlobalISel call lowering.
FastISel deliberately bails out to SelectionDAG for ``throws`` calls
(``X86FastISel.cpp``). See :ref:`herbceptions` for the per-target
mechanism (carry flag on x86/AArch64/ARM via the ADD-with-AllOnes trick in
``LowerReturn``; extra return value on RISC-V/LoongArch guarded by
``ArgFlags.isThrows()``; extra multivalue result on WebAssembly).
``GetReturnInfo`` skips zero-sized return parts for ``throws`` functions
(``llvm/lib/CodeGen/TargetLoweringBase.cpp``) and sets the ISD ``Throws``
argument flag. When ``supportThrowsCC()`` is false the discriminant is a
regular struct member.

The IR-level ``{ union{T, E}, i1 }`` shape uses the trailing ``i1`` as a
**CF placeholder** on targets where ``supportThrowsCC()`` is true. The
frontend's ``arrangeLLVMFunctionInfo`` always emits the ``i1`` as a
struct member (``getDirect({union, i1})``); the backend then pattern-
matches the post-call ``i1`` extraction to ``HERB_SETCCr`` (X86) or
equivalent on other targets, which reads the carry flag and folds it
into the ``i1`` -- so the ``i1`` never consumes a register or memory
slot. When ``supportThrowsCC()`` is false, the ``i1`` is a real struct
member carried in the next available register or stack slot.

Win64 expanded ABI
------------------

The standard Win64 ABI returns scalars only in RAX and passes types larger
than 8 bytes by pointer. This conflicts with herbceptions because the
discriminant lives in the carry flag (CF), requiring the payload to reside
in registers.

The following changes apply only to functions with the ``throws``
attribute; non-``throws`` functions follow the standard Win64 ABI
unchanged.

**Empty structs.** An empty struct (zero-sized) does not consume a register
slot for the return value. ``GetReturnInfo`` skips zero-sized types when the
``throws`` attribute is present (``llvm/lib/CodeGen/TargetLoweringBase.cpp``).

**Return values (RAX+RDX).** ``CanLowerReturn`` returns ``true`` for Win64
``throws`` functions when every non-discriminant return part is at most i64
after decomposition. The standard calling convention (``RetCC_X86``) already
assigns i64 leaves to ``[RAX, RDX, RCX, R8]`` via ``RetCC_X86Common``, so
the two i64 halves of a 16-byte payload naturally land in RAX and RDX. The
i1 discriminant is carried in CF via the ADD-with-AllOnes trick and never
consumes a register.

**Parameters (RCX+RDX / R8+R9 / stack).** The ``WinX86_64ABIInfo`` frontend
ABI classifier (``clang/lib/CodeGen/Targets/X86.cpp``) is extended with an
``IsThrows`` flag threaded through ``computeInfo`` (derived from
``FI.getHerbceptionErrorType() != nullptr``). When the function carries a
herbception error type, record types that are exactly 16 bytes and have no
destructor (``isDestructedType() == DK_none``) are coerced to
``{i64, i64}`` instead of being passed by pointer. ``ComputeValueTypes``
decomposes the coerced struct into two i64 leaves; each leaf independently
consumes one of the four integer parameter registers (RCX, RDX, R8, R9), or
spills to the stack. This matches the i686 Windows fastcall pattern where
two i32 values split into ECX+EDX.

Types with a destructor, types not exactly 16 bytes, and non-power-of-two
sizes continue to follow the standard Win64 rule (passed by pointer).

**Frame pointer.** Win64 ``throws`` functions force a frame pointer
(``X86ISelLoweringCall.cpp``) so the epilogue uses ``MOV RSP, RBP`` (which
does not touch EFLAGS) instead of ``ADD RSP, imm`` (which clobbers CF before
the ``ret``).

**New calling convention definitions.** ``RetCC_X86_Win64_C_Throws`` and
``CC_X86_Win64_C_Throws`` are documented in
``llvm/lib/Target/X86/X86CallingConv.td``. They mirror the standard Win64
conventions but serve as explicit, named entry points for the expanded
register set.

Linker: LTO ODR checking
========================

The herbception specifier is deliberately not part of the C++ mangled name,
so a ``throws`` function and its plain counterpart share the same symbol.
Two translation units that disagree about the specifier (or about a
``return_failure{E}`` error type) therefore link silently, and calls compiled against
the wrong ABI read garbage return values. Only the linker sees all of the
definitions, so LTO diagnoses this.

``llvm/include/llvm/LTO/Config.h`` defines ``lto::HerbceptionODRChecker``,
shared by every module of an LTO link through
``Config::HerbceptionODR`` (a ``shared_ptr`` so ThinLTO backend threads all
observe the same registry; controlled by ``Config::CheckHerbceptionODR``,
on by default). For every externally visible function of each materialized
bitcode module, ``checkHerbceptionODRForModule`` (``llvm/lib/LTO/LTO.cpp``)
records whether it carries the IR ``throws`` attribute and the payload
element type of its ``{T, i1}`` return. Regular-LTO modules are scanned in
``LTO::addRegularLTO``; ThinLTO modules are scanned on the backend threads
right after parsing. Conflicts are reported when ``LTO::run`` finishes::

    ld.lld: error: herbception ODR violation: symbol '_Z3fooi' has
    conflicting definitions: it is defined as a herbception ('throws')
    function with error payload type '{ ptr, i64 }' in 'a.o', but without
    the herbception error channel in 'b.o'

Modules that are never materialized in-process -- ``--thinlto-index-only``,
ThinLTO cache hits and out-of-process DTLTO backends -- are not scanned,
and definitions coming from native relocatable files are out of reach.

Runtime: libherbceptions
========================

``libherbceptions/`` in the monorepo:

* ``include/herbceptions/error`` -- a single header providing:

  - ``class std::error``: default/copy/move construction and assignment all
    deleted; ``constexpr ~error()`` runs the domain's ``do_cleanup``;
    accessors ``domain()``, ``code()``, ``equivalent(T)``, ``to_errc()``,
    ``throw_dynamic_exception()`` (rethrows as a traditional exception via
    the domain vtable, falling back to ``std::system_error(to_errc())``),
    ``is_code_of<T>()``; private two-word payload
    ``{error_domain_singleton const *, std::size_t}`` and a private magic
    constructor only the compiler uses.
  - ``struct error_domain_singleton`` with members ``do_cleanup``,
    ``do_equivalent``, ``do_query_information`` (writev-style scatter/gather
    name/message query with requested encoding), ``do_to_errc`` and
    ``do_throw_dynamic_exception``.
  - the ``template <typename T> class std::error_domain;`` customization
    point (declared, never defined; any specialization may be thrown).
  - ``namespace std::error_domains``: extern ``"C"`` singletons
    ``__cxa_error_domain_{posix,win32,nt,com,wine,cmath,parse}()``.
  - ``operator==`` between ``std::error`` and any domained value,
    ``herbception_cast``, and ``std``-style trait aliases for the compiler
    builtin traits, gated on ``__HERBCEPTIONS__``.
* ``include/herbceptions/__details/{posix,win32,nt,com,wine,cmath_errc,
  parse,exception_ptr}.h`` -- per-domain helpers. ``exception_ptr.h``
  defines the ``error_domain<std::exception_ptr>`` specialization delegating
  to the itanium/msvc entry points; this header is what makes the
  legacy-EH conversion available (see `Whole-function conversion`_).
* ``src/`` -- one translation unit per domain: ``posix.cpp``, ``win32.cpp``,
  ``nt.cpp``, ``com.cpp``, ``wine.cpp``, ``cmath_errc.cpp``,
  ``parse.cpp``, plus the legacy-EH bridges ``itanium_exception_ptr.cpp`` /
  ``msvc_exception_ptr.cpp`` (which own the
  ``__cxa_error_domain_*_exception_ptr`` / ``__cxa_error_code_*_exception_ptr``
  symbols consumed directly by compiler-fabricated code), shared query
  helpers (``simple_query_information_common.h``,
  ``__malloc_or_heap_alloc_temp_buffer.h``), the NTSTATUS tables
  (``ntkernel.h``, ``nt_message_table.hpp``, ``nt_errc_map.hpp``) and an
  EBCDIC table.
* ``test/domain_test.cpp`` -- domain identity, ``do_to_errc``,
  cross-domain equivalence, ``do_query_information`` output.
* ``fuzz/`` and ``utils/`` -- fuzzers and table generators.

Build integration: ``libherbceptions/CMakeLists.txt`` offers
``HERBCEPTIONS_BUILD_SHARED/STATIC/FREESTANDING/TESTS/FUZZERS`` options, and
the runtime is registered in ``runtimes/CMakeLists.txt``.

Testing in Clang/LLVM
=====================

Compiler behavior is covered by ``clang/test/`` files run with
``-fherbceptions``:

* Sema: ``herbception-catch-fails.cpp``,
  ``herbception-c-bare-call.c``, ``herbception-constexpr.cpp``,
  ``herbception-constexpr-throws.cpp``,
  ``herbception-coroutine-throws.cpp`` (coroutine rejection),
  ``herbception-domain-nullptr.cpp``, ``herbception-dtor-spec.cpp``,
  ``herbception-fails-trivially-copyable.cpp``, ``herbception-fnptr.cpp``,
  ``herbception-legacy-convert-no-domain.cpp``,
  ``herbception-throws-noexcept.cpp``, ``herbception-traits.cpp``.
* CodeGen: ``herbception-autoprop.cpp`` (auto-propagation, ``catch return_failure``),
  ``herbception-catch-throws.cpp`` and
  ``herbception-catch-throws-autoprop.cpp`` (block handlers),
  ``herbception-catch-fails.cpp``, ``herbception-coroutine.cpp`` (a
  non-throws coroutine under ``-fherbceptions``),
  ``herbception-fails-noexcept.cpp`` (terminate vs. propagate),
  ``herbception-legacy-convert.cpp`` (the ``catch throws(std::error)``
  legacy conversion on Itanium/MSVC/Wasm/SjLj),
  ``herbception-throws.cpp`` (``{T, i1}`` lowering, ``try(expr)``),
  ``herbception-two-field-struct.cpp`` (the ``{T, i1}`` heuristic).
* Preprocessor: ``herbceptions-macro.cpp`` (``__HERBCEPTIONS__``).

LLVM-side tests: ``llvm/test/Feature/throws-attr.ll`` (attribute
round-trip), backend tests
``llvm/test/CodeGen/{X86,AArch64,ARM,RISCV,LoongArch,WebAssembly}/
throws-attr.ll`` plus the x86 frame-pointer/CFI variants
(``throws-cfi-fp.ll``, ``throws-cfi-no-fp.ll``), and the TableGen test
``llvm/test/TableGen/callingconv-ifthrows.td``.

Known limitations
=================

* ``FastISel`` falls back to SelectionDAG for ``throws`` calls.
* Legacy-EH conversion requires the ``libherbceptions`` runtime ABI symbols
  and visible ``std::exception_ptr`` / ``std::error`` declarations; when a
  legacy escape is possible without them, compilation return_failure rather than
  silently skipping the conversion.
* ``RetCC_X86_Win64_C_Throws`` / ``CC_X86_Win64_C_Throws`` are selected
  automatically when a function carries the ``Throws`` attribute on Win64
  targets, via a ``CCIfThrows`` predicate in ``X86CallingConv.td``.

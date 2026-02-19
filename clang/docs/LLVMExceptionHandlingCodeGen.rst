========================================
LLVM IR Generation for EH and Cleanups
========================================

.. contents::
   :local:

Overview
========

This document describes how Clang's LLVM IR generation represents exception
handling (EH) and C++ cleanups. It focuses on the data structures and control
flow patterns used to model normal and exceptional exits, and it outlines how
the generated IR differs across common ABI models.

For details on the LLVM IR representation of exception handling, see
`LLVM Exception Handling <https://llvm.org/docs/ExceptionHandling.html>`_.

Core Model
==========

EH and cleanup handling is centered around an ``EHScopeStack`` that records
nested scopes for:

- **Cleanups**, which run on normal control flow, exceptional control flow, or
  both. These are used for destructors, full-expression cleanups, and other
  scope-exit actions.
- **Catch scopes**, which represent ``try``/``catch`` handlers.
- **Filter scopes**, used to model dynamic exception specifications and some
  platform-specific filters.
- **Terminate scopes**, used for ``noexcept`` and similar termination paths.

Each cleanup is a small object with an ``Emit`` method. When a cleanup scope is
popped, the IR generator decides whether it must materialize a normal cleanup
block (for fallthrough, branch-through, or unresolved ``goto`` fixups) and/or an
EH cleanup entry (when exceptional control flow can reach the cleanup). This
results in a flattened CFG where cleanup lifetime is represented by the blocks
and edges that flow into those blocks.

Key Components
==============

The LLVM IR generation for EH and cleanups is spread across several core
components:

- ``CodeGenModule`` owns module-wide state such as the LLVM module, target
  information, and the selected EH personality function. It provides access to
  ABI helpers via ``CGCXXABI`` and target-specific hooks.
- ``CodeGenFunction`` manages per-function state and IR building. It owns the
  ``EHScopeStack``, tracks the current insertion point, and emits blocks, calls,
  and branches. Most cleanup and EH control flow is built here.
- ``EHScopeStack`` is the central stack of scopes used to model EH and cleanup
  semantics. It stores ``EHCleanupScope`` entries for cleanups, along with
  ``EHCatchScope``, ``EHFilterScope``, and ``EHTerminateScope`` for handlers and
  termination logic.
- ``EHCleanupScope`` stores the cleanup object plus state data (active flags,
  fixup depth, and enclosing scope links). When a cleanup scope is popped,
  ``CodeGenFunction`` decides whether to emit a normal cleanup block, an EH
  cleanup entry, or both.
- Cleanup emission helpers implement the mechanics of branching through
  cleanups, threading fixups, and emitting cleanup blocks.
- Exception emission helpers implement landing pads, dispatch blocks,
  personality selection, and helper routines for try/catch, filters, and
  terminate handling.
- ``CGCXXABI`` (and its ABI-specific implementations such as
  ``ItaniumCXXABI`` and ``MicrosoftCXXABI``) provide ABI-specific lowering for
  throws, catch handling, and destructor emission details.
- The cleanup and exception handling code generation is driven by the flow of
  ``CodeGenFunction`` and its helper classes traversing the AST to emit IR for
  C++ expressions, classes, and statements.

AST traversal in ``CodeGenFunction`` emits code and pushes cleanups or EH scopes,
``EHScopeStack`` records scope nesting, cleanup and exception helpers materialize
the CFG as scopes are popped, and ``CGCXXABI`` supplies ABI-specific details for
landing pads or funclets.

Cleanup Destination Routing
===========================

When multiple control flow exits (``return``, ``break``, ``continue``,
fallthrough) pass through the same cleanup, the generated IR shares a single
cleanup block among them. Before entering the cleanup, each exit path stores a
unique index into a "cleanup destination" slot. After the cleanup code runs, a
``switch`` instruction loads this index and dispatches to the appropriate final
destination. This avoids duplicating cleanup code for each exit while preserving
correct control flow.

For example, if a function has both a ``return`` and a ``break`` that exit
through the same destructor cleanup, both paths branch to the shared cleanup
block after storing their respective destination indices. The cleanup epilogue
then switches on the stored index to reach either the return block or the
loop-exit block.

When only a single exit passes through a cleanup (the common case), the switch
is unnecessary and the cleanup block branches directly to its sole destination.

Branch Fixups for Forward Gotos
-------------------------------

A ``goto`` statement that jumps forward to a label not yet seen poses a special
problem. The destination's enclosing cleanup scope is unknown at the point the
``goto`` is emitted. This is handled by emitting an optimistic branch and
recording a "fixup." When the cleanup scope is later popped, any recorded fixups
are resolved by rewriting the branch to thread through the cleanup block and
adding the destination to the cleanup's switch.

Exceptional Cleanups and EH Dispatch
====================================

Exceptional exits (``throw``, ``invoke`` unwinds) are routed through EH cleanup
entries, which are reached via a landing pad or a funclet dispatch block,
depending on the target ABI.

For Itanium-style EH (such as is used on x86-64 Linux), the IR uses ``invoke``
to call potentially-throwing operations and a ``landingpad`` instruction to
capture the exception and selector values. The landing pad aggregates any
catch and cleanup clauses for the current scope, and branches to a dispatch
block that compares the selector to type IDs and jumps to the appropriate
handler.

For Windows, LLVM IR uses funclet-style EH: ``catchswitch`` and ``catchpad`` for
handlers, and ``cleanuppad`` for cleanups, with ``catchret`` and ``cleanupret``
edges to resume normal flow. The personality function determines how these pads
are interpreted by the backend.

Personality and ABI Selection
=============================

Each function with exception handling constructs is associated with a
personality function (e.g. __gxx_personality_v0 for C++ on Linux). The
personality function determines the ABI-specifc EH behavior of the
function. The IR generation selects a personality function based on language
options and the target ABI (e.g., Itanium, MSVC SEH, SJLJ, Wasm EH). This
decision affects:

- Whether the IR uses landing pads or funclet pads.
- The shape of dispatch logic for catch and filter scopes.
- How termination or rethrow paths are modeled.
- Whether certain helper functions such as exception filters must be outlined.

Because the personality choice is made during IR generation, the CFG shape
directly reflects ABI-specific details.

Example: Array of Objects with Throwing Constructor
===================================================

Consider:

.. code-block:: c++

  class MyClass {
  public:
    MyClass(); // may throw
    ~MyClass();
  };
  void doSomething(); // may throw
  void f() {
    MyClass arr[4];
    doSomething();
  }

High-level behavior
-------------------

- Construction of ``arr`` proceeds element-by-element. If an element constructor
  throws, destructors must run for any elements that were successfully
  constructed before the throw in reverse order of construction.
- After full construction, the call to ``doSomething`` may throw, in which case
  the destructors for all constructed elements must run, in reverse order.
- On normal exit, destructors for all elements run in reverse order.

Codegen flow and key components
-------------------------------

- The surrounding compound statement enters a ``CodeGenFunction::LexicalScope``,
  which is a ``RunCleanupsScope`` and is responsible for popping local cleanups
  at the end of the block.
- ``CodeGenFunction::EmitDecl`` routes the local variable to
  ``CodeGenFunction::EmitVarDecl`` and then ``CodeGenFunction::EmitAutoVarDecl``,
  which in turn calls ``EmitAutoVarAlloca``, ``EmitAutoVarInit``, and
  ``EmitAutoVarCleanups``.
- ``CodeGenFunction::EmitCXXAggrConstructorCall`` emits the array constructor
  loop. While emitting the loop body, it enters a ``RunCleanupsScope`` and uses
  ``CodeGenFunction::pushRegularPartialArrayCleanup`` to register a
  cleanup before calling ``CodeGenFunction::EmitCXXConstructorCall`` for one
  element in the loop iteration. If this constructor were to throw an exception,
  the cleanup handler would destroy the previously constructed elements in
  reverse order.
- ``CodeGenFunction::EmitAutoVarCleanups`` calls ``emitAutoVarTypeCleanup``,
  which ultimately registers a ``DestroyObject`` cleanup via
  ``CodeGenFunction::pushDestroy`` / ``pushFullExprCleanup`` for the full-array
  destructor path.
- ``DestroyObject`` uses ``CodeGenFunction::destroyCXXObject``, which emits the
  actual destructor call via ``CodeGenFunction::EmitCXXDestructorCall``.
- Cleanup emission helpers (e.g., ``CodeGenFunction::PopCleanupBlock`` and
  ``CodeGenFunction::EmitBranchThroughCleanup``) thread both normal and EH exits
  through the cleanup blocks as scopes are popped.
- The cleanup is represented as an ``EHCleanupScope`` on ``EHScopeStack``, and
  its ``Emit`` method generates a loop that calls the destructor on the
  initialized range in reverse order.

The above function names and flow are accurate as of LLVM 22.0, but this is
subject to change as the code evolves, and this document might not be updated to
reflect the exact functions used.

Example: Temporary object materialization
=========================================

Consider:

.. code-block:: c++

  class MyClass {
  public:
    MyClass();
    ~MyClass();
  };
  void useMyClass(MyClass &);
  void f() {
    useMyClass(MyClass());
  }

High-level behavior
-------------------

- The temporary ``MyClass`` is materialized for the call argument.
- The temporary must be destroyed at the end of the full-expression, both on
  the normal path and on the exceptional path if ``useMyClass`` throws.
- If the constructor throws, the temporary is not considered constructed and no
  destructor runs.

Codegen flow and key functions
------------------------------

- ``CodeGenFunction::EmitExprWithCleanups`` wraps the full-expression in a
  ``RunCleanupsScope`` so that full-expression cleanups are run after the call.
- ``CodeGenFunction::EmitMaterializeTemporaryExpr`` creates storage for the
  temporary via ``createReferenceTemporary`` and initializes it. For record
  temporaries this flows through ``EmitAnyExprToMem`` and
  ``CodeGenFunction::EmitCXXConstructExpr``, which calls
  ``CodeGenFunction::EmitCXXConstructorCall``.
- ``pushTemporaryCleanup`` registers the destructor as a full-expression
  cleanup by calling ``CodeGenFunction::pushDestroy`` for
  ``SD_FullExpression`` temporaries.
- The cleanup ultimately uses ``DestroyObject`` and
  ``CodeGenFunction::destroyCXXObject``, which emits
  ``CodeGenFunction::EmitCXXDestructorCall``.

The above function names and flow are accurate as of LLVM 22.0, but this is
subject to change as the code evolves, and this document might not be updated to
reflect the exact functions used.

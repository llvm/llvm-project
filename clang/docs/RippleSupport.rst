.. contents::
   :local:

==================================
Ripple Language Extensions Support
==================================

Clang has full support for the vectorization and loop annotation API exposed by Ripple.

TODO: add link to publicly available API reference manual.

Internals Manual
================

This section acts as internal documentation for Ripple features design
as well as some important implementation aspects. It is primarily targeted
at the advanced users and the toolchain developers integrating frontend
functionality as a component.

Ripple API Support
------------------

Clang exposes most of the Ripple API through internal headers:

- ``ripple.h``: Core API and pragma definitions.
- ``ripple_math.h``: Math builtins.
- ``ripple_hvx.h``: HVX-specific helpers.
- ``ripple/zip.h``: Tuple unzipping and block loading utilities.
- ``__ripple_vec.h``: High-level C/C++ interfaces for vectorization and builtins.

Ripple Builtins
---------------

Clang exposes the Ripple API as core (target-independent) builtins:

- Builtins are defined in ``BuiltinsRipple.td`` and integrated via ``Builtins.td``.
- These include:
  - Core: ``__builtin_ripple_get_index``, ``__builtin_ripple_get_size``, ``__builtin_ripple_set_shape``, ``__builtin_ripple_parallel_idx``.
  - Reductions: ``__builtin_ripple_reduceadd_*``, ``__builtin_ripple_reducemax_*``, ``__builtin_ripple_reducemin_*``, ``__builtin_ripple_reduceand_*``, ``__builtin_ripple_reduceor_*``.
  - Math: ``__builtin_ripple_sqrt``, ``__builtin_ripple_exp``, ``__builtin_ripple_log``, etc.
  - Shuffle/Broadcast: ``__builtin_ripple_shuffle_*``, ``__builtin_ripple_broadcast_*``, ``__builtin_ripple_slice_*``.

Ripple Semantics Analysis
-------------------------

Ripple constructs are semantically analyzed via the ``SemaRipple`` class:

- Validates canonical loop form for Ripple parallel constructs.
- Ensures dimension indices are unique.
- Checks builtin argument types and values.
- Constructs ``RippleComputeConstruct`` AST nodes for transformed loops.

Ripple AST Integration
----------------------

Ripple introduces a new AST node ``RippleComputeConstruct`` defined in ``StmtRipple.h``:

- Represents a parallel loop construct with full and remainder loop sections.
- Includes source ranges, block shape reference, dimension indices, and sub-statements.
- Supports traversal via ``RecursiveASTVisitor`` and printing via ``StmtPrinter``.

Ripple Builtin Infrastructure
-----------------------------

Ripple builtins are defined in ``BuiltinsRipple.td`` using TableGen templates:

- Integer and floating-point types are supported via type templates.
- Builtins are lowered to LLVM intrinsics in ``CGStmtRipple.cpp``.
- Semantic checks are performed in ``SemaRipple::CheckBuiltinFunctionCall``.

Ripple Code Generation
----------------------

Ripple constructs are lowered to LLVM IR via ``EmitRippleComputeConstruct``:

- Emits full iteration loop over blocks.
- Emits remainder loop if needed.
- Generates runtime condition and loop variable updates.
- Maps builtins to LLVM intrinsics.

Ripple Driver and Language Options
----------------------------------

Ripple support is enabled via:

- ``-fenable-ripple``: Enables Ripple transformation.
- ``-fripple-lib=<path>``: Adds a file to the Ripple bitcode library path.
- ``-fdisable-ripple-lib``: Disables use of Ripple bitcode libraries.

These options are defined in ``Options.td`` and handled in ``Clang.cpp``.

Ripple Diagnostics
------------------

Ripple introduces new diagnostics:

- ``warn_pragma_ripple_ignored``: Unexpected `#pragma ripple`.
- ``err_ripple_loop_not_for_loop``: Ripple only applies to `for` loops.
- ``err_ripple_loop_not_canonical_*``: Validates loop initializer, condition, increment.
- ``err_ripple_duplicate_parallel_index``: Duplicate dimension index.
- ``err_ripple_stmt_depends_on_loop_counter``: Loop expressions depend on control variable.

Defined in ``DiagnosticParseKinds.td`` and ``DiagnosticSemaKinds.td``.

Ripple Header Files
-------------------

Ripple headers are installed via ``lib/Headers/CMakeLists.txt``:

- ``ripple.h``, ``ripple_math.h``, ``ripple_hvx.h``, ``__ripple_vec.h``, ``ripple/zip.h``.
- Provide high-level interfaces for vectorization, math, and parallel constructs.

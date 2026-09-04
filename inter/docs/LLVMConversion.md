# LLVM Import and XW Conversion

## Role

Inter has two distinct LLVM boundaries:

1. `inter-translate --import-llvm` uses upstream translation to create MLIR
   LLVM dialect IR from native LLVM IR.
2. The backend pipeline imports kernel ABI facts, canonicalizes supported
   frontend ABI shims, structures control flow, and performs a full conversion
   from LLVM dialect to XW.

This document describes accepted current behavior. Checked-in O2/O3 examples
are witnesses, not a claim that arbitrary optimized LLVM is supported.

## Native LLVM import

`inter-translate --import-llvm` parses LLVM IR and calls upstream
`translateLLVMIRToModule`. Parsing and translation failures are fatal. Inter
does not independently require a particular optimization pipeline or verify an
exact `spir64` triple at this entry point.

The resulting module retains LLVM data layout for ABI and GEP lowering.

## Kernel import

`inter-import-llvm` converts each defined `llvm.func` to `func.func`.

Defined functions must:

- use the `spir_kernel` calling convention;
- be non-variadic;
- have only integer, floating-point, or LLVM pointer arguments.

Defined helpers are rejected and must be inlined before import. External
declarations remain temporarily for recognized builtin conversion.

The imported function preserves its symbol, visibility, body, and non-LLVM
discardable attributes. It receives:

- `xw.kernel`;
- `xw.simd_width`, selected as 8, 16, or 32;
- `xw.kernel_args`, the ordered argument descriptor array.

## Kernel argument descriptors

Argument payload placement begins at byte 24. The closest MLIR data layout
provides size and ABI alignment. Every descriptor records:

- byte offset, size, and alignment;
- by-value or pointer kind;
- pointer address space;
- pointer access mode: read-only, write-only, or read-write.

Offsets are aligned and monotonically packed. The selector and zebin emitter
consume the same descriptor array; neither recomputes placement from argument
index.

The descriptor does not currently retain arbitrary LLVM parameter attributes,
`noalias`, dereferenceability, TBAA, or alias scopes.

## Control-flow normalization

Kernel import rewrites LLVM branches and returns to `cf` and `func` operations.
Conditional branch weights are copied to the temporary CF branch, but upstream
CFG lifting does not preserve them in SCF. The pipeline then runs:

1. `lift-cf-to-scf`;
2. canonicalization;
3. `inter-prepare-counted-loops`;
4. `inter-verify-structured`.

Counted-loop preparation recognizes a narrow signed comparison/add recurrence
shape and exposes it to SCF lifting. Any remaining LLVM or CF branch fails the
structured boundary. Irreducible and otherwise unsupported CFGs are not carried
into XW.

## Frontend ABI canonicalization

Before full LLVM conversion, two passes recognize exact Intel builtin ABI forms.

### Cache controls

`inter-discover-cache-controls` parses supported pointer annotations and follows
them through GEP, casts, freeze, select, and block arguments. Exact profiles
become `xw.cache_control` on consumers. Conflicting annotated paths or merges of
annotated and unannotated state are hard errors.

### Block2D builtins

Exact mangled signatures are recognized for:

- 16-bit 8x16 prefetch;
- 16-bit 8x16 read;
- 16-bit 16x16 transformed read;
- 32-bit 8x16 write.

The canonicalizer validates the private alloca shim, coordinate construction,
use set, and load/store ordering, then creates XW block2D operations and removes
dead ABI scaffolding.

### DPAS builtins

Mangled Intel subgroup matrix-mad names are parsed for f16 or bf16 source
precision and positive K. Calls require three arguments, one fixed vector
result, and an enclosing SIMD width. They become `xw.dpas` operations.

These recognizers are exact ABI conversion, not general contraction recovery.

## Type conversion

Implemented mappings include:

- LLVM pointers in spaces 0 through 4 to opaque XW private, global, constant,
  local, and generic pointers;
- LLVM arrays to fixed one-dimensional builtin vectors;
- non-opaque LLVM structs to builtin tuples;
- non-variadic LLVM function types to builtin function types;
- LLVM integers and bytes to signless integers;
- fixed vectors by recursive element conversion;
- LLVM void to `none`.

Type support does not imply support for every operation manipulating that type.
General aggregate operations are not accepted merely because aggregate layout
is available for GEP conversion.

## Accepted operation subset

### Values and integer arithmetic

Supported forms include LLVM constants, pointer zero, full poison, freeze,
integer add/subtract/multiply, signed/unsigned division and remainder, shifts,
and bitwise operations. Integer overflow flags are preserved where XW declares
them. LLVM undef and partial poison are rejected.

### Floating point

Floating add, subtract, and multiply are accepted and become full-width XW SIMD
operations with explicit splats for bare operands. Floating division and
remainder are rejected. LLVM fast-math flags are not comprehensively preserved
by the current conversion.

### Casts, comparisons, and select

Supported casts include integer extension/truncation, floating extension/
truncation, integer/float conversion, bitcast, address-space cast, ptr-to-int,
and int-to-ptr. Local/generic address-space casts are rejected.

All non-pointer integer predicates and implemented floating predicates are
converted. Pointer comparison supports equality and inequality only. Selects
reconcile bare and SIMD arms through explicit shape conversion.

### Loads, stores, and atomics

Loads and stores become token-producing XW memory operations. A load result is
conservatively full-width regardless of pointer uniformity.

Volatile and atomic LLVM loads/stores are rejected because ordinary XW
load/store operations have no exact representation for those semantics.

Atomic RMW currently accepts only nonvolatile monotonic integer add with no
syncscope. LLVM fences and unsupported atomic operations/orderings fail.

Explicit LLVM alignment remains a source promise rather than a durable XW
attribute. Atomic ordering and scope are represented only by the supported
`atomic_rmw` subset.

## GEP conversion

LLVM GEP becomes byte-oriented `xw.ptradd`. Source element types and MLIR data
layout are consumed to compute strides and struct padding.

Supported traversal includes:

- the first source-element index;
- LLVM arrays;
- fixed builtin vectors;
- non-opaque LLVM structs with constant in-range field indices.

Dynamic indices are converted to the address-space index width. Local pointers
use 32-bit offsets. Power-of-two strides may be expanded as repeated addition;
other strides use multiplication. SIMD indices produce SIMD pointers.

Scalable indexed types, dynamic struct fields, unknown address widths, and
unsupported aggregates fail. GEP `inbounds` and no-wrap flags are not preserved
as XW contracts today.

## Globals and local memory

Referenced address-space-3 globals become packed local-memory allocations. Size
and alignment come from data layout; module order determines aligned offsets.
`llvm.mlir.addressof` becomes `xw.local_memory_base`, and the local global is
removed.

Referenced non-local globals and scalable local globals are rejected. General
global, constant, and private storage lowering is not implemented.

## Builtins

Recognized direct calls include subgroup/local IDs, subgroup ID, global/local/
group IDs, launch and workgroup size queries, barrier, and atomic add. Dimension
queries require a constant axis from 0 through 2.

Indirect calls and unknown direct calls fail. External declarations are erased
only after all recognized uses have been converted.

## Structured XW conversion

`inter-convert-llvm-to-xw` uses full dialect conversion.

- Uniform `scf.if` remains SCF.
- A mask condition creates `xw.where`.
- Loop-carried values are reconciled to common bare/SIMD shapes.
- A lane-varying `scf.while` condition is rejected.
- Function signatures, block arguments, operation results, and nested
  attributes are converted consistently.

After conversion and shape reconciliation, Inter explicitly rejects:

- every LLVM dialect operation;
- every CF operation;
- every unrealized conversion cast;
- every LLVM type in operands, results, or attributes.

This closed-boundary check is the contract that permits downstream passes to
ignore LLVM representation details.

## Metadata preservation

Currently preserved or materialized facts include:

- data layout while ABI, GEP, and local-memory layout are computed;
- symbol visibility and non-LLVM discardable attributes;
- conditional branch weights until CFG-to-SCF lifting;
- supported integer overflow flags;
- resolved cache controls;
- kernel identity, SIMD width, and argument descriptors.

The current implementation does not promise durable preservation of arbitrary
LLVM alias metadata, TBAA, access groups, loop metadata, fast-math, volatility,
alignment, dereferenceability, complete atomic semantics, or LLVM-native
analysis results. Module attributes beginning with `llvm.` and the data-layout
attribute are removed after they have served the conversion.

## Pipeline position

The authoritative order is in `lib/inter/pipelines/pipelines.mlir`:

```text
import -> annotation/builtin canonicalization -> structure CFG
  -> full LLVM-to-XW conversion -> distribution refinement
  -> arithmetic legalization -> memory-token inference -> selection
```

## Current limits

- Defined helpers must already be inlined.
- Only the checked-in LLVM operation subset is accepted.
- General VP/masked operations, reductions, shuffles, and contraction recovery
  are absent.
- CFG structuring handles only supported reducible forms.
- Imported LLVM alias/range analyses are not materialized into XW.
- Generic pointer provenance and arbitrary globals are unsupported.
- The accepted O2/O3 corpus is evidence, not universal optimizer compatibility.

## Normative sources and tests

- Native import: `tools/inter-translate/inter-translate.cpp`
- Kernel import: `lib/inter/Transforms/InterImportLLVM.cpp`
- Conversion: `lib/inter/Transforms/InterConvertLLVMToXW.cpp`
- Builtin canonicalization: `lib/inter/Transforms/InterCanonicalizeBlock2DABI.cpp`
- Cache controls: `lib/inter/Transforms/InterDiscoverCacheControls.cpp`
- Counted-loop preparation: `lib/inter/Transforms/InterPrepareCountedLoops.cpp`
- Structured verification: `lib/inter/Transforms/InterImportLLVM.cpp`
- Positive tests: `test/Frontend/` and `test/Transforms/convert-llvm-to-xw*.mlir`
- Rejection tests: `test/Transforms/convert-llvm-to-xw-reject*.mlir`

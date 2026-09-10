# XW Frontend Dialect

## Role

XW is Inter's closed semantic SIMD dialect. It is the first IR that no longer
depends on LLVM operations or LLVM types. XW represents kernel execution,
lane-distributed values, opaque pointers, structured control flow, memory
effects, and semantic ordering without committing to physical Intel registers
or SWSB encodings.

The ODS definitions and handwritten verifiers under
`include/inter/Dialect/Inter/IR` and `lib/inter/Dialect/Inter/IR` are normative.

## Execution scope and value model

An XW kernel carries `xw.simd_width`, currently 8, 16, or 32.

One kernel subgroup maps to one EU hardware thread. For a logical lane `L`,
`!xw.simd<T, N>` uses the contiguous compact mapping
`floor(L * N / W)`, where `W` is the enclosing SIMD width. Non-contiguous lane
partitions require explicit movement or remain full-width.

- Bare builtin values are uniform across the hardware subgroup.
- `!xw.simd<T, N>` stores `N` lane-distributed values.
- `!xw.mask<N>` stores a predicate with cardinality `N`.
- `N` must be positive and divide the enclosing SIMD width.
- Mixed cardinalities require an operation-specific promotion rule or explicit
  `xw.expand`; arbitrary implicit redistribution is forbidden.

Builtin vectors are per-lane packets, not subgroup lanes. For example,
`!xw.simd<vector<8xf16>, 16>` contains one eight-element packet per SIMD16 lane.

The current type system records cardinality but not a matrix-fragment layout.
DPAS A/B packing is therefore a contract between producers such as block2D
reads and `xw.dpas`, rather than a distinct fragment type.

## Types and attributes

### Pointers

`!xw.ptr<#space>` is opaque and carries only an address space:

- private
- global
- constant
- local
- generic

Pointer arithmetic is byte-oriented. Pointee types are consumed during LLVM GEP
conversion and do not survive in XW.

### Memory tokens

`!xw.mem.token` represents semantic ordering. It is distinct from
`!xemachine.mem.token` and from physical SBIDs. Memory operations consume an
optional dependency and return a token.

### Operation policy attributes

XW defines closed enums for integer operations, numeric casts, cache policies,
and f16/bf16 DPAS precision. Standard MLIR overflow and fast-math attributes are
used where declared.

## Core operations

### Values and packets

- `xw.constant` creates bare, SIMD, or mask constants.
- `xw.splat` distributes a bare value.
- `xw.read_first` returns one value from a SIMD input.
- `xw.expand` increases compact cardinality by an exact integral factor.
- `xw.freeze` preserves supported LLVM freeze semantics.
- `xw.pack` concatenates equal-shaped values into a fixed vector packet.
- `xw.extract` selects a constant scalar or contiguous packet slice.
- `xw.bitcast` is legal only when total bit width and SIMD shape agree.

There is no proof-bearing `xw.compact`, general redistribution operation, or
general reduction operation today.

### Arithmetic and conversion

`xw.binary` covers the implemented integer/index operations. Bare/bare operands
produce a bare value; one SIMD operand promotes the result to that cardinality;
two SIMD operands must agree. `xw.cast` records signedness, extension, rounding,
and overflow policy explicitly.

Floating operations include add, subtract, multiply, maximum, FMA, exp2, and
reciprocal in the dialect. Selection currently implements only a subset and
fails on unsupported operations rather than approximating them.

### Predicates

- `xw.cmpi`, `xw.cmpf`, and `xw.ptr_cmp` support bare comparisons returning
  `i1` and SIMD comparisons returning `!xw.mask<N>`.
- Mask boolean operations preserve cardinality.
- `xw.select` accepts either a uniform `i1` or a compatible mask.
- `xw.ballot` materializes mask bits in i8, i16, or i32.

### Builtins and cross-lane operations

XW models lane, subgroup, work-item, workgroup, and launch queries. Query axes
are constant 0, 1, or 2. `xw.shuffle` preserves SIMD shape and accepts a bare or
matching-cardinality source-lane operand, although current selection supports
only constant lanes and payloads no wider than 32 bits.

## Structured control flow

Uniform control uses `scf.if`, `scf.for`, and `scf.while`. Lane-varying
conditionals use `xw.where` with `xw.yield`.

`xw.where` regions are single-block, have no block arguments, and yield exactly
the operation result types. A result-bearing operation requires an else region.
Bare values crossing a SIMD boundary are explicitly splatted during conversion.

Lane-varying loop conditions are not represented. LLVM conversion rejects an
`scf.while` condition that becomes a mask.

## Memory model

### Ordinary memory

- `xw.load` and `xw.store` use bare or SIMD pointers and explicit value shapes.
- `xw.atomic_rmw` represents the supported atomic operation surface.
- `xw.local_memory_base`, `xw.alloc`, and `xw.alloc_release` represent local
  memory addresses and lifetimes.
- `xw.barrier` consumes explicit dependencies and returns a token.

Pointer and value cardinalities must agree when the pointer is SIMD. A bare
pointer does not imply a bare load result.

### Block2D

`xw.block2d_prefetch`, `xw.block2d_read`, and `xw.block2d_write` carry uniform
surface geometry, coordinates, element width, block dimensions, transpose/VNNI
flags, data, and dependency tokens.

The base must be a bare global or constant pointer. Geometry is uniform i32.
Writes require one untransformed block; transformed reads require one block.
Data is a SIMD value containing a fixed one-dimensional vector packet.

### DPAS

`xw.dpas` carries A, B, accumulator, source precisions, K, systolic depth, and
repeat count. A/B/accumulator/result must be SIMD values containing fixed
one-dimensional packets. The accumulator and result types match, K equals two
times systolic depth for the supported source precisions, and repeat count
matches the result packet length.

The verifier checks packet shape and size, not semantic A/B packing. The current
Lighthouse path obtains A directly from an ordinary block2D read and B directly
from a VNNI-transformed block2D read.

## Token algebra and inference

- `xw.token` creates a root.
- `xw.issue_token`, `xw.after`, and `xw.join` merge dependencies.
- Memory operations preserve explicit dependencies and receive inferred alias
  dependencies before selection.
- Tokens are threaded through if/where results and loop-carried state.

The three merge operations currently share simple folding behavior; the type
system does not encode separate issue and completion token classes.

`inter-infer-memory-tokens` uses MLIR alias analysis and memory effects. Reads
depend on potentially aliasing earlier writes. Writes depend on potentially
aliasing reads and writes. Read/read pairs need no edge. Barriers conservatively
order all relevant chains. Block2D prefetch dependencies are deferred to the
next potentially aliasing read.

Physical completion is not an XW concern. Selection lowers these tokens to
machine ordering tokens, and post-allocation SWSB insertion handles register
readiness.

## Distribution refinement

The implemented dataflow lattice is cardinality only:

```text
uninitialized -> bare or simd<N>
```

Joins use least common multiple, bounded by the function SIMD width. Constants,
uniform queries, token operations, and local-memory bases are bare. Lane/global/
local IDs and ordinary load values conservatively begin at full width.

After convergence, `inter-refine-distribution` updates structural equivalence
classes spanning branch results, yields, loop arguments, and backedges. A
candidate refinement is rolled back if operation verification fails. The pass
does not refine function signatures and cannot derive affine patterns such as
`lane_id >> 1`.

## Boundary invariants

A valid closed XW module has:

- No LLVM dialect operation or LLVM type.
- No CF dialect operation.
- No unrealized conversion cast.
- XW plus the supported `func`/`scf` structure, selected `arith` operations,
  structural terminators, and full `ub.poison` values accepted by selection.
- Structured region yields and loop carries with exact compatible types.
- SIMD and mask cardinalities valid for the enclosing width.
- Explicit memory dependencies after token inference.

Unsupported operations, pointer spaces, cardinality combinations, control-flow
forms, or memory nesting are hard errors.

## Current limitations

- XW includes Intel-specific block2D and DPAS operations.
- No dedicated matrix fragment/layout types exist.
- Distribution analysis tracks cardinality, not affine lane expressions.
- No compact, assume, redistribute, gather/scatter, or general reduction ops.
- Atomics and barriers do not expose the full LLVM ordering/scope model.
- Semantic token inference supports only single-block functions and supported
  single-block structured regions around memory operations.
- Token inference uses current MLIR alias analysis, not imported MemorySSA,
  TBAA, alias scopes, or runtime alias checks.

## Normative sources and tests

- Types and operations: `include/inter/Dialect/Inter/IR/`
- Verifiers/folders: `lib/inter/Dialect/Inter/IR/XWDialect.cpp` and
  `lib/inter/Dialect/Inter/IR/XWOps.cpp`
- Distribution: `lib/inter/Analysis/DistributionAnalysis.cpp` and
  `lib/inter/Transforms/InterRefineDistribution.cpp`
- Tokens: `lib/inter/Transforms/InterInferMemoryTokens.cpp`
- Dialect tests: `test/Dialect/Inter/`
- Distribution and token tests: `test/Analysis/distribution.mlir` and
  `test/Analysis/memory-tokens*.mlir`

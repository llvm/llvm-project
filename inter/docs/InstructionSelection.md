# Instruction Selection

## Role and boundary

`inter-select-to-machine` converts a closed XW kernel and supported standard
structured operations into XeMachine. Selection decides register packet shapes,
machine operations, message payloads and descriptors, structured machine control
flow, and the kernel payload prologue. It does not assign physical registers,
schedule instructions, or infer SWSB.

The implementation in `lib/inter/Transforms/InterSelect.cpp` is normative.

## Kernel construction

Selection finds functions marked as XW or XeMachine kernels. A kernel must have
no results and one outer block. Selection creates a new argumentless machine
function, transfers the symbol, and records:

- BMG target;
- SIMD width;
- 128 selected GRFs and five reserved GRFs;
- required workgroup size;
- kernel argument descriptors;
- payload and local-memory metadata.

Function arguments become payload reads. Function return emits EOT using the
current machine memory token.

## Value representation

The selector maps each semantic value to machine storage, immediate, flag, or
token values.

- Register width is measured in dwords.
- Bare values have cardinality one.
- SIMD cardinality comes from XW types.
- Index values are 64-bit.
- Local pointers are 32-bit offsets.
- Global, constant, and generic pointers are 64-bit.
- SIMD32 64-bit values and A64 pointers are represented as paired SIMD16
  packets where required.

Constants become immediates when encodable. Packet splats materialize whole-GRF
pieces. Pack/extract become tuple storage views. Bitcasts are zero-cost only
when source and result footprints match.

## Arithmetic and predicates

Implemented integer operations map directly or expand to available machine ALU
operations. XOR is synthesized from OR, AND, and subtraction. Multiplication
uses the accumulator and moves the result to GRF storage.

Implemented floating selection includes add, subtract, and multiply. Floating
maximum, FMA, exp2, reciprocal, and unsupported predicates fail rather than
changing semantics.

Comparisons produce flag ARFs. Mask boolean operations use flag-compatible
machine operations, and ballot moves flag bits into integer storage. Uniform
and mask selects lower to structured machine conditionals with yielded values.

Only the explicitly implemented subset of SIMD32 i64 arithmetic is accepted.

## Kernel payload and arguments

Kernel argument descriptors must be ordered, aligned, nonoverlapping, and fit
within the current 192-byte payload window. Explicit arguments begin at byte 24.
Pointer arguments are eight bytes with concrete address-space and access fields.

- Bytes below 32 are read from the inline payload register.
- Bytes 32 through 191 are loaded in aligned 64-byte chunks.
- An argument may not cross one loaded-tail chunk.
- Bare arguments are extracted through scalar no-mask moves.

When local IDs are needed, selection emits `payload_prologue`:

1. copy inline data to its common-body register;
2. derive the indirect per-thread payload address from architectural state;
3. load required local-ID axes into fixed registers;
4. mark the alternate walker entry with `payload_prologue_end`;
5. drain prologue sends on the common path after the region.

The common body then sees one register layout regardless of whether software or
hardware local-ID generation was used. The hardware-derived contract is in
[PayloadContract.md](PayloadContract.md).

## IDs and cross-lane operations

Selection currently supports local and global IDs, group ID, local size, launch
block size, subgroup ID, and constant-lane shuffle.

- Local IDs are loaded as i16 and widened.
- Global IDs combine local ID, group ID, local size, and payload global offset.
- Subgroup ID linearizes local IDs and divides by SIMD width.
- Shuffle requires a constant lane and a payload no wider than 32 bits.

Lane ID, dynamic shuffle, global size, number of groups, and launch grid size
currently fail because the required machine primitive or payload field is not
implemented.

## Ordinary memory

Selection supports 32-bit memory elements.

- Local pointers use SLM load/store messages with 32-bit addresses.
- Global, constant, and generic pointers use A64 UGM messages.
- SIMD32 A64 addresses are joined from two SIMD16 halves.
- Atomic selection supports only i32 A64 add.
- Local allocation/base values become immediate SLM offsets.
- Allocation release forwards its dependency token without emitting memory.

Private memory access and unsupported element widths fail.

## Block2D selection

Block2D operations build one 16-dword address payload:

| Dword | Field |
|---|---|
| 0-1 | 64-bit base |
| 2 | surface width minus one |
| 3 | surface height minus one |
| 4 | pitch minus one |
| 5 | X coordinate |
| 6 | Y coordinate |
| 7 | packed block width, height, and count minus one |
| 8-15 | zero |

Current raw descriptors are:

- prefetch: `0x02080203`;
- ordinary read: `0x02400203`;
- VNNI-transformed read: `0x02800283`;
- write: `0x02000407`.

Prefetch returns an issue-only `xemachine.after` token. Reads return register
packets and completion-bearing tokens.

The transformed read performs B operand VNNI packing in the memory message. An
ordinary 8x16 FP16 read produces the four-GRF DPAS A/source2 packet; a transformed
16x16 FP16 read produces the eight-GRF B/source1 packet. No separate relayout is
inserted before DPAS.

The XW verifier accepts a broader semantic geometry surface than the current
fixed descriptor mapping. BMG selection explicitly rejects combinations outside
the four forms above. There is no generic message fallback chain today.

## DPAS selection

Selection materializes A, B, and accumulator packets and creates
`xemachine.dpas`. F16 and BF16 source precision map directly; accumulation is
f32. The machine result destructively aliases the accumulator.

For the Lighthouse depth-8/repeat-8 form:

```text
ordinary block2D A read -> 64 dwords -> DPAS source2
transformed block2D B read -> 128 dwords -> DPAS source1
float8-per-lane accumulator -> 128 dwords -> accumulator/result
```

Storage bitcasts from the builtin ABI are removed during canonicalization. The
selector does not infer or repair matrix layout. Producers must satisfy the DPAS
packet contract.

## Barriers and tokens

XW token operations become machine token, after, or token-join operations.
Memory dependencies must already be explicit.

A barrier lowers to:

1. SLM fence;
2. fence-await;
3. gateway barrier payload construction;
4. barrier signal;
5. `sync.bar`.

The selector tracks the current machine token so EOT and barrier sequences
retain outstanding side-effect ordering.

## Structured control flow

- `xw.where` becomes `xemachine.exec_if`.
- `scf.if` becomes `xemachine.uniform_if`.
- `scf.for` becomes `xemachine.uniform_loop` with explicit induction and
  carried state.
- Supported `scf.while` becomes a uniform loop with a nested uniform conditional
  and explicit state forwarding.

Region tokens are ordinary yielded or carried values. Malformed asymmetric
state, incompatible result storage shapes, or surviving unsupported control
flow fails selection.

## Post-selection canonicalization

Before scheduling:

- `inter-coalesce-tuples` factors equivalent update templates and includes a
  descriptor-local preference for destinationless reads.
- `inter-reuse-block2d-payloads` reuses invariant payloads and encodes legal
  immediate X/Y deltas in exdesc.
- LICM hoists legal invariant machine operations.
- register-allocation preparation later repairs any unsafe destructive or
  relative aliasing introduced by these forms.

## Hard failures

Selection rejects missing kernels, kernel results, unstructured outer CFG,
invalid ABI descriptors, unsupported types/operations, non-splat constants,
footprint-changing bitcasts, unsupported wide arithmetic, private memory,
unsupported memory widths, invalid pointer casts, unsupported atomics and
builtins, malformed loops, and any unrecognized remaining operation.

There is no silent generic fallback below this boundary.

## Current limits

- Target configuration currently resolves only BMG/Xe2 with 128 GRFs.
- Ordinary memory supports dword elements.
- Block2D descriptors cover the canonicalized Lighthouse forms.
- DPAS supports f16/bf16 source packets and f32 accumulation.
- Matrix layout is implicit in producer contracts.
- Several XW floating and builtin operations are deliberately unselected.
- No general gather/scatter or generic contraction fallback is implemented.

## Normative sources and tests

- Selector: `lib/inter/Transforms/InterSelect.cpp`
- Builtin canonicalization: `lib/inter/Transforms/InterCanonicalizeBlock2DABI.cpp`
- Tuple/payload transforms: `lib/inter/Transforms/InterCoalesceTuples.cpp` and
  `lib/inter/Transforms/InterReuseBlock2DPayloads.cpp`
- Selection tests: `test/Transforms/select-*.mlir`
- DPAS and block2D tests: `test/Transforms/select-dpas.mlir`,
  `test/Transforms/select-block2d.mlir`, and `test/Emit/dpas.mlir`
- ABI tests: `test/Transforms/kernel-args-invalid.mlir` and
  `test/Emit/zebin.mlir`

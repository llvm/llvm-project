# XeMachine Backend Dialect

## Role

XeMachine is Inter's final MLIR representation of Intel EU machine code. It
exists from instruction selection through scheduling, register allocation,
synchronization, resource analysis, and emission. There is no lower MLIR
dialect: XeMachine lowers to a buffered C++ emission program and then GED.

The ODS definitions under `include/inter/Dialect/XeMachine/IR` and handwritten
verification in `lib/inter/Dialect/XeMachine/IR` are normative.

## Storage types

### GRFs

`!xemachine.reg<widthDwords, baseGRF>` describes contiguous general-register
storage. One Xe2 GRF is 16 dwords or 64 bytes.

- Base `-1` is virtual.
- A nonnegative base is physical.
- Width zero represents a null destination.
- Allocation rewrites the base field in place; there is no separate physical
  register type.

### ARFs

`!xemachine.arf<file, widthDwords, index>` represents architectural register
storage including `a0`, flags, accumulators, MME, and special files. Negative
indices are virtual. Current allocation supports only virtual flag registers;
other ARFs must already be physical.

### Immediates and tokens

`!xemachine.imm` carries an SSA immediate whose producer records the element
type. `!xemachine.mem.token` records machine issue ordering and completion
bookkeeping independently of physical SWSB fields.

## Kernel metadata

Function attributes carry target and ABI inputs:

- target chip
- kernel argument descriptors
- SIMD and required workgroup size
- selected and reserved GRF counts
- inline and per-thread payload sizes
- local-memory and scratch sizes

`xemachine.target` accepts checked textual construction or a typed target-chip
value; there is no unchecked string builder. The immutable target configuration
resolves the chip before selection and owns its architecture, GRF geometry,
supported SIMD widths, and Zebin compatibility identity. BMG is the only
accepted chip today; unknown chips and feature requests fail before selection
mutates a kernel.

`KernelABI` separately owns source address spaces, pointer representation,
explicit and implicit argument layout, cross-thread and per-thread payload
geometry, reserved payload registers, and scratch conventions. Hardware GRF
geometry remains a target property; the five-register payload reservation is
an ABI property.

After physical lowering, resource analysis adds highest GRF use, barrier presence,
global atomic use, stateless-write status, and DPAS use. Zebin emission
recomputes and cross-checks these derived facts.

## Instruction representation

### ALU

ALU operations carry execution size, mask offset, `noMask`, element type,
destination/source regions, subregisters, optional source type overrides, and
final SWSB fields. Implemented operations include move, add/subtract, shifts,
logic, add3, multiply through the accumulator, and compare.

Absent regions use canonical scalar/contiguous defaults. Region legality is an
encoding contract, not an optimization hint.

### Sends

Raw `xemachine.send` stores the shared function, descriptors, execution fields,
address payload, optional data payload, optional register exdesc, dependency,
destination, and output token.

Named message operations cover A64 and SLM loads/stores, A64 atomic add, A32
block load, SLM fence and await, barrier signal, and EOT. Named-message encoding
constants live in emission; raw block2D sends currently retain descriptors
selected earlier.

### DPAS

`xemachine.dpas` is an asynchronous SIMD16 systolic instruction. Its semantic
operand order is A, B, accumulator. Emission maps B to hardware source 1 and A
to source 2. The destination destructively aliases the accumulator.

For Xe2 depth 8 and repeat 8, the required packets are:

- A: 64 dwords / 4 GRFs.
- B: 128 dwords / 8 GRFs.
- accumulator/result: 128 dwords / 8 GRFs.

The verifier enforces f32 accumulation, depth 8, repeat 1 through 8, and packet
widths. Virtual destination/accumulator aliasing is represented through the
storage-alias interface; the verifier checks equal bases once both are physical.
It does not identify whether A/B bytes carry the required semantic matrix layout.

### Synchronization

Every SWSB-capable operation stores:

- distance pipe and distance
- scoreboard token
- token mode: set, source wait, or destination wait

`xemachine.sync` represents `nop`, `allrd`, `allwr`, and `bar`. Selective
`allwr` carries an SBID mask. These fields are final after synchronization
insertion. Emission serializes them, while defensively supplying the mandatory
first direct `a0` floating-pipe distance if hand-authored IR omitted it.

## Zero-byte operations and storage aliases

`tuple_from_elements`, `tuple_to_elements`, and `update_tuple` expose contiguous
storage relationships without copying bytes. Tuple elements occupy whole GRFs;
updates may target sub-GRF offsets. Physical placements must match the declared
relative offsets.

`xemachine.dpas` also exposes the destructive accumulator/result alias.
Structured region transfers add zero-offset aliases between operands, block
arguments, yields, and results.

`RegisterStorageAliasOpInterface` makes these relations available to the
scheduler, preparation, and allocator. Weighted dword offsets form immutable
alias components after preparation.

Other zero-byte operations include token creation/merging, `after`, fixed
register references, and payload-prologue markers. Traits distinguish operations
that emit no instruction, have no completion obligation, or require a full
scoreboard drain.

## Structured machine control flow

### Payload prologue

`payload_prologue` contains software local-ID setup. The alternate walker entry
bypasses the region and starts at `payload_prologue_end` with the same fixed
register contract.

### Conditionals

- `exec_if` models divergent mask-stack control and emits goto/join structure.
- `uniform_if` models subgroup-uniform control. At divergent depth zero it can
  emit predicated jumps; nested cases use structured goto/join emission.

### Loops

`uniform_loop` has explicit initial, body, backedge, and result values and ends
with `continue_if`. At emission, all non-token instances of one carried value
must occupy identical physical storage.

Region flow, repetition, reachability, and mutually exclusive alternatives are
shared by scheduling and register-allocation preparation through
`XeMachineRegionFlow`.

## Interfaces

Important operation interfaces are:

- `ALUOpInterface` for regions, types, and execution controls.
- `SWSBInfoOpInterface` for final synchronization fields.
- `AsyncScoreboardOpInterface` for sends and DPAS.
- `RegisterStorageAliasOpInterface` for relative storage.
- `InstructionIssueOpInterface` for target timing classification.
- `RegionLegalityOpInterface`, `SendDescriptorOpInterface`, and
  `FixedPhysicalRegisterDefsOpInterface` for machine constraints.

Emission still uses explicit operation dispatch. There is no generic emission
interface that permits unknown machine operations to pass through.

## Phase invariants

### Before scheduling

- Semantic memory ordering is represented by token SSA.
- Destructive and region aliases have been prepared.
- Machine operations may still use virtual GRFs and flags.

### Before allocation

- Instruction order is fixed.
- Alias components and machine operands are legal for allocation.

### Before synchronization

- All GRFs and supported ARFs are physical.
- Instruction order will not change again.
- Message and DPAS packet footprints are verifier-valid.

### Before emission

- Every encodable instruction has final SWSB metadata.
- Resource attributes match physical IR.
- No unsupported operation remains.

## Emission

`XeMachineLowering.cpp` translates known operations to an `EmissionProgram`
containing instructions, labels, and branch fixups. GED emits fixed 16-byte Xe2
instructions. Structured regions become labels and `goto`, `jmpi`, or `join`.
Zero-byte operations are omitted.

The zebin writer currently emits one BMG kernel with `.text`, symbols, zeinfo,
and IntelGT compatibility notes. It requires 128-GRF mode and validates payload
and resource metadata.

## Current limitations

- Only BMG/Xe2 is implemented.
- Instruction compaction is absent.
- Zebin emission supports one kernel and 128-GRF mode.
- Raw descriptor constants remain in block2D selection and payload reuse.
- DPAS supports f16/bf16 inputs to f32 with Xe2 depth 8.
- Matrix fragment layout is not encoded in machine types.
- No independent final physical-hazard simulator validates SWSB annotations.
- Several interface hooks remain conservative defaults rather than complete
  per-operation legality models.

## Normative sources and tests

- Definitions: `include/inter/Dialect/XeMachine/IR/`
- Verifiers and region flow: `lib/inter/Dialect/XeMachine/IR/`
- Emission: `lib/inter/Emit/`
- Dialect tests: `test/Dialect/XeMachine/`
- Emission tests: `test/Emit/`
- Hardware payload evidence: [PayloadContract.md](PayloadContract.md)

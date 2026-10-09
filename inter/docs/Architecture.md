# Inter Architecture

## Status and scope

This document describes the implemented Inter compiler architecture. It is the
entry point for the subsystem documents in this directory; operation definitions,
verifiers, the checked-in transform pipeline, and tests remain normative when
prose and code disagree.

Inter compiles a supported subset of optimized SPIR64 LLVM IR to native Intel
Xe2 code packaged as zebin:

```text
LLVM IR
  -> MLIR LLVM dialect
  -> XW semantic SIMD dialect
  -> XeMachine virtual-register dialect
  -> scheduled and physically allocated XeMachine
  -> synchronized XeMachine
  -> GED instruction bytes
  -> zebin ELF
```

The initial target is Battlemage G21, validated on Arc Pro B60. The device path
does not use IGC, vISA, or SPIR-V. Inter links Intel GED for instruction encoding
and emits zebin directly.

## Compilation API

`inter::compileLLVMModule` is the tool-independent compilation boundary. It
owns LLVM-to-MLIR import, dialect and pass registration, execution of the
canonical transform library, diagnostic capture, and final emission. Callers
provide an owned LLVM module, a validated `TargetConfig`, SIMD width, transform
library path, and an output kind: zebin, raw GED bytes, IGA assembly, or
validation-only. A caller-owned stream receives MLIR diagnostics, and the API
returns `llvm::Error`; it does not terminate the process or depend on global
command-line options.

`inter-opt` uses the same compiler dialect and pass registration functions, and
`inter-translate` uses the same GED, assembly, and zebin emitters. The transform
library remains the sole declaration of backend pass order.

## Design principles

- Every IR boundary is printable and verifier-backed.
- Unsupported input fails at a declared boundary; no selector case disappears
  silently.
- Semantic memory ordering is explicit before machine scheduling.
- Register allocation finishes before physical synchronization is inferred.
- Instruction order is frozen before SWSB annotation.
- Emission serializes final decisions and does not repair scheduling,
  allocation, or general synchronization. The one defensive exception is the
  required first direct `a0` floating-pipe distance, which emission verifies and
  fills when hand-authored machine IR omitted it.
- Hardware-specific facts are centralized in XeMachine, Xe2 timing, emission,
  and the hardware payload contract.

## Document map

- [FrontendDialect.md](FrontendDialect.md): XW types, operations, distribution,
  structured control, and semantic tokens.
- [BackendDialect.md](BackendDialect.md): XeMachine storage, instructions,
  aliases, control flow, and emission contract.
- [LLVMConversion.md](LLVMConversion.md): accepted LLVM input and the closed
  LLVM-to-XW boundary.
- [InstructionSelection.md](InstructionSelection.md): XW-to-XeMachine lowering,
  payload construction, block2D, DPAS, memory, and control flow.
- [Scheduling.md](Scheduling.md): dependency construction, Xe2 timing, pressure,
  and the pre-allocation gap-filling scheduler.
- [RegisterAllocation.md](RegisterAllocation.md): alias preparation, ARF/GRF
  allocation, relief, scratch, and physical validation.
- [Synchronization.md](Synchronization.md): semantic memory-token inference and
  physical SWSB insertion.
- [PayloadContract.md](PayloadContract.md): hardware-observed BMG payload and
  message facts used by selection and zebin emission.

## Executable pipeline

`lib/inter/pipelines/pipelines.mlir` is the authoritative pass order. The
`inter_backend` entry point performs the following stages.

The complete pipeline is invoked with:

```text
--pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=<pipelines.mlir>},transform-interpreter{entry-point=inter_backend})'
```

### LLVM preparation

1. `inter-import-llvm` identifies kernels and records their ABI descriptors.
2. `inter-discover-cache-controls` resolves supported pointer annotations.
3. Block2D ABI shims and subgroup matrix builtins become semantic XW operations.
4. CFG lifting, canonicalization, and counted-loop preparation form supported
   structured control flow.
5. `inter-verify-structured` rejects surviving unstructured branches.

### Semantic XW

1. `inter-convert-llvm-to-xw` performs a full conversion and rejects surviving
   LLVM operations, LLVM types, CF operations, and unrealized casts.
2. Distribution refinement narrows safe lane cardinalities.
3. Integer narrowing, arithmetic expansion, canonicalization, and CSE finish
   semantic legalization.
4. `inter-infer-memory-tokens` creates explicit source-level memory ordering.

### Machine lowering

1. `inter-select-to-machine` creates an argumentless XeMachine kernel with an
   explicit payload prologue and machine operations.
2. Tuple coalescing and block2D payload reuse optimize storage views and message
   payload construction.
3. LICM runs before physical preparation.

### Physical pipeline

1. `inter-prepare-regalloc` repairs destructive aliases, tuple placement, and
   structured transfers.
2. `inter-machine-schedule` performs the only machine scheduling pass.
3. ARF allocation assigns virtual flags.
4. The GRF transform loop repeatedly builds allocation state, runs linear scan,
   and tries rematerialization then scratch spilling after failure.
5. `inter-insert-sync` allocates SBIDs, inserts waits, and adds ALU distance
   dependencies after physical register assignment.
6. `inter-resource-info` validates physical bounds and publishes resource
   metadata used by zebin emission.

There is currently no post-allocation repair scheduler. Instructions inserted by
allocation relief proceed directly to synchronization.

## Ownership boundaries

### XW owns source semantics

XW distinguishes uniform and lane-distributed values, opaque pointer address
spaces, structured lane predication, memory effects, and semantic dependency
tokens. Alias-based memory ordering is decided here. Later passes consume token
SSA and do not rediscover source-level aliasing.

### XeMachine owns target representation

XeMachine owns virtual and physical GRF/ARF storage, EU regions, send payloads
and descriptors, structured machine control flow, DPAS packets, storage aliases,
and final SWSB fields. Zero-byte tuple and token operations remain visible to
analysis but do not emit instructions.

### Scheduler owns order, not policy

The generic scheduler builds dependencies and chooses from a legal ready set.
The Xe2 model owns timing, pipes, filler compatibility, physical-storage hazards,
and pressure policy. Semantic memory ordering reaches the scheduler only through
SSA token edges.

### Register allocation owns placement

Preparation makes destructive and relative-placement constraints explicit.
Allocation assigns whole-GRF component bases and the supported ARFs. It must
finish before SWSB because synchronization operates on physical register spans.

### Synchronization owns physical readiness

SWSB insertion tracks asynchronous source-read and destination-completion
obligations, assigns hardware scoreboard IDs, materializes waits, and computes
ALU distance annotations. It does not change instruction order.

### Emission owns serialization

`XeMachineLowering.cpp` creates a buffered emission program. GED emits fixed
16-byte Xe2 instructions; zebin emission writes code, symbols, zeinfo, and
compatibility notes. The emitter checks required metadata but performs no
dependency inference or register allocation.

## Kernel ABI and resources

Kernel import computes one ordered argument descriptor list using MLIR data
layout. Selection consumes it to read inline and indirect payload fields, while
zebin emission consumes the same offsets, sizes, alignments, address spaces, and
access modes.

The current BMG kernel contract uses:

- SIMD8, SIMD16, or SIMD32 execution.
- 128 GRFs with five reserved GRFs in selected kernels.
- A 32-byte inline payload.
- Optional 64, 128, or 192-byte per-thread local-ID payload.
- One software-local-ID entry and one walker-provided local-ID entry.
- Exactly one kernel per emitted zebin.

`inter-resource-info` derives highest GRF use, barrier presence, global atomic
use, DPAS use, and stateless-write status. The zebin writer recomputes these
facts and rejects stale attributes.

## Emission and runtime

`inter-translate` provides:

- `--xemachine-to-ged` for native instruction bytes.
- `--xemachine-to-asm` for GED-backed disassembly.
- `--xemachine-to-zebin` for a native device image.

Integration tests use the repository's runner infrastructure, while the
Lighthouse benchmark uses Level Zero directly for timestamped compiler
comparison. `PayloadContract.md` records the payload facts established by B60
register-dump probes.

The integration runner wraps raw zebin in LLVM's `OffloadBinary` container with
image kind `IMG_Object` and a `spirv64` triple. It validates and loads that image
through liboffload's Level Zero backend with `olIsValidBinary` and
`olCreateProgram`; the plugin then passes the native payload to Level Zero.

## Validation model

- Dialect tests exercise parsing, verification, folding, and invalid IR.
- Transform tests specify each analysis and rewrite boundary.
- Emission tests check GED fields, disassembly, resource metadata, and zebin.
- Integration tests launch serialized hardware workloads on B60.
- Performance goldens freeze reviewed machine assembly.
- The Lighthouse benchmark rebuilds Inter and the pinned Lighthouse
  MLIR-to-XeVM pipeline, validates dense randomized matrix products, and
  compares Level Zero kernel timestamps.

Hardware-test instructions are in `test/Integration/README.md`; benchmark
methodology is in `benchmarks/README.md`.

## Current limits

- Target metadata and timing are implemented only for BMG/Xe2.
- LLVM conversion supports a deliberate subset, not arbitrary optimized LLVM.
- XW contains Intel block2D and DPAS semantic operations despite otherwise
  serving as the frontend boundary.
- Matrix operand layout is currently a producer/consumer contract, not a
  dedicated fragment type.
- Machine allocation is whole-GRF; sub-GRF placement is represented only by
  alias offsets and instruction subregisters.
- Only virtual flag ARFs are allocated.
- There is no live-range splitting or post-relief scheduling.
- Emission supports 128-GRF mode, one kernel, and uncompressed 16-byte
  instructions.
- No independent post-SWSB physical-hazard simulator exists; synchronization is
  tested directly and through hardware stress instead.

These are implementation boundaries, not fallback promises. Unsupported cases
must continue to fail explicitly until their complete contracts are added.

## Normative sources

- Pipeline: `lib/inter/pipelines/pipelines.mlir`
- Pass declarations: `include/inter/Transforms/Passes.td`
- XW: `include/inter/Dialect/Inter/IR/`
- XeMachine: `include/inter/Dialect/XeMachine/IR/`
- Selection and physical passes: `lib/inter/Transforms/`
- Register allocation: `lib/inter/Dialect/XeMachine/IR/XeMachineRegAlloc*.cpp`
- Emission: `lib/inter/Emit/`
- Tests: `test/Analysis`, `test/Transforms`, `test/Emit`, `test/Integration`, and
  `test/PerfGolden`

External ground truths are:

- Intel GED from the IGC revision pinned by `cmake/FetchGED.cmake` for Xe2
  instruction encoding and decoding.
- IGC `visa/LocalScheduler/LatencyTable.*` as the source adapted by the local
  Xe2 timing model.
- Compute Runtime's zebin decoder under
  `third_party/compute-runtime/shared/source/device_binary_format/zebin` for ELF
  and zeinfo compatibility.
- The DG2/Alchemist PRM as a payload proxy, cross-checked by the B60 observations
  in `PayloadContract.md`.
- wave-mlir commit `a0bef6698ceb8eda58d44c113c9616b2317c7bb3` as structural
  inspiration for the machine dialect, transform-loop allocation, scheduler
  split, and pipeline-as-data organization.

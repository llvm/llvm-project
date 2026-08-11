# inter: Intel Xe2 GPU Backend in MLIR — Design

Status: draft. Project root: `inter/` in this repo. First target: Xe2 / Battlemage
G21 (Arc Pro B60, PCI 8086:e211), Linux, `xe` KMD, Level Zero runtime.

## 1. Goal

A compiler for verified, optimized SPIR64 LLVM IR:

```
LLVM IR after the normal LLVM -O2/-O3 pipeline
  -> MLIR LLVM dialect
  -> semantic Inter SPMD/tile IR
  -> xemachine dialect
  -> Intel EU binary
  -> zebin ELF
  -> Level Zero
```

No IGC, no vISA, no SPIR-V, no NEO compiler interfaces on the device path.
The host side uses LLVM's `liboffload` (in this repo) over Level Zero as
module loader and runtime; Inter makes no direct `ze*` calls, and liboffload's
L0 plugin loads the emitted image through
`zeModuleCreate(ZE_MODULE_FORMAT_NATIVE)`. All device-side knowledge below the
machine dialect — encoding, scoreboarding, container format — is owned here.

Correctness before performance. Every pipeline stage is printable, inspectable
MLIR. No pass keeps hidden C++ state across an IR rewrite. Optimized LLVM forms
are accepted by semantics, not by matching one kernel, one argument order, one
GEP shape, one unroll factor, or one optimization level.

## 2. Non-goals for the production-matmul milestone

- Instruction compaction (64-bit forms). The milestone emits 128-bit
  instructions only.
- Function calls, relocations, indirect branches, SIP/debug support.
- Ray tracing, media fixed functions, bindless-only kernels.
- Multiple Intel platforms. Everything is Xe2; arch gating exists but only
  `xe2` is populated.
- Automatic discovery of arbitrary tensor programs. The milestone recognizes
  semantically valid contraction recurrences in optimized LLVM and exact
  subgroup matrix intrinsics; unrelated algorithms remain on the generic path.
- Beating every vendor-tuned GEMM shape. The first release must be correct,
  general over its declared types/layouts/tails, and measurably use block
  messages and DPAS. Tuning breadth follows correctness.

XMX/DPAS, matrix fragment layouts, SIMD16, 2D block load/store, constrained
register allocation, reusable SWSB tokens, and production matmul validation are
goals, not deferred work. Both 128- and 256-GRF modes belong to the target cost
model; the initial accepted kernel may choose 128 GRFs when it fits.

## 3. Ground truths

The ISA and container are fully determined by open sources. Nothing here
requires reverse engineering.

| What | Where | Role |
|---|---|---|
| EU encoding, Xe2 bit-exact | `intel/intel-graphics-compiler`, `visa/iga/GEDLibrary` (GED) | encoder/decoder library, MIT, C API; XE2 tables annotated per-field in `ged_enumerations.h` |
| Xe2 scheduling timing | same repo, `visa/LocalScheduler/LatencyTable.*` | static completion latency, issue occupancy, send-source read time, and dependency-gap policy |
| Reference assembler/disassembler | same repo, `visa/iga` (IGA CLI and `iga.h`) | `iga -p xe2 -a/-d`; verification oracle |
| Container format | `intel/compute-runtime`, `shared/source/device_binary_format/zebin/` | zebin decoder; ground truth for ELF + `.note.zeinfo` |
| Reference backend | Mesa `src/intel/compiler/elk_*` | RA, SWSB scheduling, regioning legality, payload setup |
| Payload / walker / shared-local memory rules | Intel graphics PRM hub, Alchemist (DG2) PRM volumes | thread payload layout, SLM sizing, execution environment. **Proxy, not truth**: no public Xe2 PRM exists; DG2 is the closest documented generation. Cross-check against elk source; see section 19 |
| Host-side launch path | this repo, `offload/` (`liboffload` + `plugins-nextgen/level_zero`) | L0 launcher for M0 and the mlir-runner shim: `olCreateProgram` -> `zeModuleCreate(ZE_MODULE_FORMAT_NATIVE)` (`L0Program.cpp`); see section 16 |
| Architecture to transplant from | wave-mlir (`github.com/Hardcode84/7`), pinned at `a0bef6698ceb8eda58d44c113c9616b2317c7bb3` | machine-dialect shape, token model, regalloc loop, scheduler split, pipeline-as-data, test tiers |

Pin commits for IGC and NEO in `inter/deps.txt` (or similar) once chosen; both
formats drift.

## 4. Intel EU execution model (the parts that shape the design)

- One hardware thread runs SIMD8/16/32 instructions. One kernel "subgroup"
  maps to one thread; a compiled SIMD width is chosen per kernel and declared
  in zeinfo (`compiled_simd_size`).
- One register bank: 128 GRFs per thread, 512-bit each on Xe2, in the
  default mode. Xe2 also has a Large GRF mode: 256 GRFs per thread at
  halved hardware threads per core. GRF mode is a per-kernel decision that
  couples the register budget to occupancy. The current prototype uses
  128-GRF mode; production tile planning chooses 128 or 256 from pressure and
  occupancy cost, and no dialect type assumes either count. Plus small ARF files:
  address registers `a0`, flags `f0/f1`, accumulators `acc`, `mme`, and specials
  (`sr0`, `cr0`, `n0`, `ip`, `tm0`, ...). No separate scalar bank; uniformity
  pays through message selection and region narrowing, not through a cheaper
  register class.
- Operands are register **regions**: sources `<V;W,H>`, destination `<H>`.
  Producer/consumer region compatibility, stride legality, and alignment
  (64-bit types even-aligned, etc.) are hard encoding constraints, not
  performance hints. This is the main way EU regalloc differs from anything
  on AMD.
- Structured control flow is architectural: `if/else/endif/while/break/cont`
  maintain an implicit execution-mask stack. Predication and `f0/f1` handle
  short divergent regions. Divergence needs no explicit reconvergence
  bookkeeping as long as emitted control flow stays structured.
- All memory and shared-function access is `send`/`sendc` with dense
  descriptors: SFID (dataport/UGM, SLM, sampler, thread spawner), message
  type, payload GRF lengths, cache-control bits. Sends are asynchronous:
  issue and completion are distinct events; the writeback lands in GRFs later.
- **SWSB.** There is no hardware scoreboard. Every instruction carries
  software scoreboard fields: token set/src/dst dependencies or sync
  functions. Wrong SWSB = silent corruption or hangs, not errors. SWSB is a
  correctness mechanism.
- Kernels are not signed or validated. The EU executes whatever the interface
  descriptor points at.
- The container is zebin: ELF64 with per-kernel `.text`, data sections, and
  `.note.zeinfo` YAML (execution environment, SLM size, barrier count, and
  `payload_arguments` describing where each kernel argument lands in the
  thread payload). NEO reads zeinfo and marshals arguments; the ABI is
  declared, not hardcoded — but must be declared exactly right.

## 5. Pipeline overview

```
verified spir64 LLVM IR after opt -O2/-O3
  | inter-translate --import-llvm
  | preserve calling convention, data layout, attributes, alias metadata,
  | fast-math/overflow flags, loop metadata, and intrinsic identity
  v
MLIR LLVM dialect
  | inter-llvm-legalize
  |   explicit conversion target: no selector fallthrough, no dropped ops
  |   preserve exact integer/pointer/poison/FP semantics
  | inter-structure-cfg
  |   reducible CFG -> structured loops/regions; retain general loop state
  v
Inter SPMD core
  | explicit uniform/lane-bundle values and active masks
  | exact byte-address expressions with provenance/range/alignment
  | semantic memory-dependence graph and async event operations
  | general scalar/fixed-vector ALU, branches, loops, gathers/scatters
  v
Inter contraction/tile level
  | recover legal contractions from optimized scalar/vector reductions
  | normalize exact subgroup matrix intrinsics to the same operation
  | choose SIMD width, M/N/K tiles, fragment layouts, tails, and staging
  | generic ALU + gather/scatter path remains available for correctness
  v
Intel semantic selection
  | legality/cost-driven 2D block -> block -> gather/scatter fallback
  | legality/cost-driven DPAS -> generic contraction fallback
  | no raw descriptor constants outside target message encoding
  v
xemachine virtual-register IR
  | software-pipeline/prefetch scheduling
  | pressure-aware machine scheduling
  | alias preparation and constrained register-allocation transform loop
  | restricted post-RA scheduling after spill/copy repair
  v
xemachine physical-register IR
  | final SWSB/SBID allocation and physical-hazard verification
  | resource-info: GRF mode, SLM, spill, barrier, DPAS, SIMD -> attrs
  v
emission (translation, not a pass)
  | GED serialization only; no dependency inference in the encoder
  | buffered fixups and branch targets
  | zeinfo YAML + ELF write
  v
zebin ELF  ->  OffloadBinary wrapper  ->  liboffload olCreateProgram
           ->  zeModuleCreate(ZE_MODULE_FORMAT_NATIVE)
```

Pipeline is data: a `transform.with_named_sequence` library interpreted by
`transform-interpreter`, same as wave-mlir's `pipelines.mlir`. All dialects
are preloaded before the interpreter runs (its multi-threaded context refuses
late registration — known pitfall, documented in wave-mlir).

The checked-in library is `lib/inter/pipelines/pipelines.mlir`, staged in the
build tree as `share/inter/pipelines/pipelines.mlir`. A complete backend run is
invoked with:

```
--pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=<pipelines.mlir>},transform-interpreter{entry-point=inter_backend})'
```

Stage discipline: one dialect mix per stage, every boundary FileCheck-able,
and every conversion uses an explicit legality target. The current direct
LLVM-to-`xemachine` selector is prototype debt and must be split at the semantic
SPMD and tile boundaries above; it is not the architecture to extend.

## 6. Frontend: LLVM IR import

- Input contract: LLVM-verified `spir64` IR after a normal target-independent
  LLVM `-O2` or `-O3` pipeline. Address spaces are private=0, global=1,
  constant=2, local=3, generic=4. Kernel entry points are identified from the
  `spir_kernel` calling convention, never from function names or from "every
  defined function". Defined non-kernel functions are helpers to inline or
  reject explicitly, not extra kernels.
- O2/O3 forms are normal input: PHIs, selects, `freeze`, fixed vectors,
  shuffles, masked/VP operations, `llvm.fma`/`llvm.fmuladd`, vector reductions,
  unrolled/peeled loops, main-loop plus epilogue, and optimizer-generated
  pointer arithmetic. The accepted subset grows through a declared conversion
  target. Any unlegalized operation is a hard diagnostic at the boundary; an
  unrecognized selector case may never silently disappear.
- Import preserves data layout and all semantic facts needed later: calling
  convention, parameter/call attributes, `noalias`, alias scopes, TBAA, access
  groups, volatility, atomic ordering/scope, alignment, dereferenceability,
  `inbounds`/no-wrap flags, fast-math flags, and loop metadata.
- LLVM-native analyses (`LoopInfo`, ScalarEvolution, AA, MemorySSA,
  LoopAccessAnalysis, and dependence analysis) run while the original CFG and
  metadata are intact. Proven recurrences, trip counts, ranges, alignments,
  disjointness, and runtime alias checks are materialized into IR attributes or
  semantic operations; they do not survive only as hidden C++ state.
- Reducible CFGs are structured without requiring one canonical optimizer
  shape. Eligible loop state becomes `scf.for`; other reducible loops retain an
  explicit general loop form. Irreducible CFGs are initially rejected with a
  precise diagnostic and later handled by a dedicated structurizer.
- The function-signature -> kernel ABI descriptor is computed from the calling
  convention, LLVM data layout, and target ABI table. It carries argument type,
  size, alignment, address space, access mode, and payload location. Zeinfo and
  prologue generation consume the same descriptor.

### Prototype debt that must be removed

The current `InterSelect.cpp` implementation is deliberately not accepted as a
general lowering path. In particular, `emitByteOffsets()` hardcodes
`global_id.x * 4`, global load/store require a single GEP whose base is a direct
block argument, `pointerArg()` derives payload locations from argument index,
SLM uses only the last GEP index times four, and selection assumes i32/SIMD32.
These shortcuts made the first hardware proof possible; matmul work deletes
them rather than adding more cases around them.

## 7. SPMD value model and uniformity

LLVM fixed-vector dimensions are per-work-item data; they are not hardware
subgroup lanes. The semantic Inter layer makes the lane axis explicit:

```
T                 // uniform across the hardware thread
simd<T, N>        // N compact values distributed across the thread
```

The enclosing kernel carries one selected hardware width `W` (for example,
`xw.simd_width = 32`). `N` is the number of stored values and must divide `W`:
`simd<T, W>` is fully lane-varying, while `simd<T, W / 2>` stores one value per
adjacent lane pair. Ordinary bare values are uniform, following wave-mlir; no
uniform wrapper type is required. Every SIMD value carries an active mask
through divergent branches, loops, tails, and masked memory operations. Lane
ID, subgroup ID, workgroup ID, local/global IDs, shuffle, broadcast, ballot,
and reduction are semantic operations before they become EU regions or control
flow.

Uniformity is dense forward dataflow above target selection. Its advisory
lattice is:

```
const  <  uniform  <  affine-strided(k, base-uniform)  <  varying
```

- `const`: compile-time known, retaining exact integer width and poison/no-wrap
  semantics.
- `uniform`: same in all lanes of the subgroup (and, where provable, the
  workgroup — needed for barrier-safety reasoning).
- `affine-strided`: lane-affine with uniform base and constant stride k.
  This is the class that selects block vs. scatter message forms; it is why
  boolean uniformity is insufficient.
- Sources: lane id (strided k=1), workgroup/subgroup IDs (uniform at their
  documented scope), constants, kernel arguments, and propagation rules
  through scalar/vector arithmetic and structured regions. Shift propagation
  uses the actual shift amount and exact modular width; the current prototype's
  "every shift doubles stride" rule is invalid and must be removed.
- The strided-ness proofs delegate to ixsimpl range/divisibility queries
  under assumptions (`wave.assume`-style: predicates attach to SSA results,
  recovered only through def chains).
- Results are consumed by: message-form selection (2D/block/gather/scatter),
  branch classification (uniform condition -> `uniform_if`/`uniform_loop`,
  else `exec_if`), and region narrowing (uniform producer allows
  `<0;1,0>`-style broadcast reads).
- Analysis results are advisory annotations, never permission to change LLVM
  semantics. Nothing below selection may re-derive uniformity; passes read the
  annotation or the explicit type. Failure to prove uniformity selects the
  varying fallback rather than rejecting a legal program.

### Typed compact SIMD values (future work)

The production semantic Inter layer should preserve uniform versus SIMD values
in types, following wave-mlir's useful rule that ordinary bare types are
uniform and only lane-distributed values use a wrapper. This is not merely a
cached uniformity result: it is a representation contract checked across
operation results, block arguments, loop-carried values, region yields, and
function boundaries.

Inter extends the Wave model by interpreting `N` in `simd<T, N>` as compact
storage cardinality under the enclosing kernel width `W`. The logical value for
lane `L` is stored element `floor(L * N / W)`, so the mapping is contiguous:

```
i32                 under W=32       // one uniform value
simd<i32, 8>         under W=32       // one value per four adjacent lanes
simd<i32, 16>        under W=32       // one value per adjacent lane pair
simd<i32, 32>        under W=32       // one value per lane
```

No separate clustered type or layout attribute is needed for this initial
model. Xe selection produces a `simd<i32, 16>` value with SIMD16 and can consume
it in SIMD32 through a source region such as `<1;2,0>`. A SIMD32 destination
must not write multiple active lanes to the same compact element; compact
values are produced at their storage width and broadcast only on reads. Sends
and other operations requiring one physical payload element per lane must
explicitly expand the value.

Elementwise operations use the least common compatible cardinality. Bare plus
bare remains bare; bare plus `simd<T, N>` produces `simd<T, N>`; and
`simd<T, A>` plus `simd<T, B>` produces `simd<T, lcm(A, B)>`, provided the
cardinality divides `W`. Coarser operands broadcast on read. Conversion from a
finer cardinality to a coarser one is never implicit and requires a proof that
the values agree in every destination group. Operations and region boundaries
that require an exact cardinality use explicit splat, expand, or proven-compact
conversion operations.

Compact SIMD representations are legal across divergent control flow only
when the active mask is compatible with the same lane grouping. If lanes that
share a stored element can have different activity while defining that value,
selection must expand before the divergent region or use `simd<T, W>`.
Uniformity and affine-stride analysis prove candidate cardinalities; typed
conversion makes the chosen representation durable through later rewrites.

This does not justify types for machine decomposition details. In particular,
an i64 remains i64 when Xe lowers its SIMD32 execution into two SIMD16
instructions. Lane distribution describes frontend value semantics and
storage cardinality; instruction splitting does not.

Exemplars to mine for lattice plumbing: `WaitLattice`/`HazardLattice` in
wave-mlir (`lib/Dialect/Wave/Transforms/WaveAMDMachineWaitcnt.cpp`,
`WaveAMDHazardWaits.cpp`) — dense dataflow over region-bearing control flow,
join/back-edge handling included.

## 8. Symbolic addressing

Every memory operation consumes:

```
opaque LLVM pointer SSA value plus the memory operation's access type,
alignment, and active mask
```

- Opaque pointers retain their LLVM address space and SSA structure through the
  semantic layer. Typed GEPs normalize to `xw.ptradd(base, byteOffset)`, where
  the offset uses the address-space-specific index width and the source element
  type plus data layout determines byte strides. GEP no-wrap/inbounds semantics
  are preserved. Loads/stores continue to consume pointers rather than a second
  target-independent address object.
- Byte-offset arithmetic preserves sign/zero extension, truncation, shifts,
  division/remainder, selects, PHI/add recurrences, `inbounds`, `nuw`/`nsw`,
  and modular integer width. Algebra may use ixsimpl or an equivalent
  hash-consed engine, but only after LLVM semantics are represented exactly.
- Pointer provenance survives GEP, bitcast, and address-space cast. Arbitrary
  ptr-to-int arithmetic is conservative; absent a proved base/range it remains
  a general address. Generic address-space pointers are specialized only when
  provenance proves a concrete space, otherwise rejected until a documented
  generic-pointer ABI exists.
- Provenance is not reconstructed when LLVM transformations have erased it.
  Range, divisibility, alignment, recurrence, and non-overlap proofs come from
  imported LLVM analysis facts plus MLIR integer-range/dataflow analysis. Passes
  query a common API and become conservative when facts are unavailable; target
  selectors do not walk arbitrary SSA looking for a favorite shape.
- Message selection evaluates each access against target capability tables and
  demotes through a correctness-preserving chain:

  ```
  legal 2D block message
    -> legal 1D/subgroup block message
    -> vector gather/scatter
    -> split or scalar masked gather/scatter
  ```

  A 2D candidate requires a proved rectangular affine lane/tile mapping, legal
  dimensions, pitch/base alignment, coordinate ranges, element type, register
  layout, transpose/VNNI mode, and out-of-bounds behavior. Dynamic legality may
  use a runtime guard plus the generic fallback. Failure to prove a block form
  never changes the address and never rejects an otherwise supported access.
- Address-space lowering: global -> A64 stateless UGM; local -> bounded SLM
  offsets; constant -> stateless read-only; private -> GRF when promoted,
  otherwise scratch. Scalar, vector, block, 2D, and prefetch operations all use
  this same pointer representation.

## 9. Memory model: tokens and early alias analysis

Memory ordering is explicit. Legality is SSA dominance plus token edges. No
pass below token synthesis may infer ordering.

- `!inter.mem.token`; memory ops take a dependency token operand and return a
  result token. `join` merges, `after` sequences without data dependence. This
  semantic token is distinct from machine send-completion/SBID state.
- Two tiers: completion tokens (data ready) and issue tokens (ordered issue,
  no completion promise). Sends consume issue-order edges; their completion
  is modeled by SWSB, not by the token graph — the token graph orders
  *issue*, the scoreboard orders *writeback*.
- Tokens thread through regions as ordinary SSA: yielded by `scf.if` /
  machine `exec_if`, carried by `scf.for` / `uniform_loop` iter args.
- **Token synthesis is the one place semantic alias analysis exists.** It runs
  above selection while LLVM `noalias`, alias scopes, TBAA, access groups,
  MemorySSA, address-space facts, and proved byte ranges remain visible.
  Distinct pointer arguments are *not* assumed disjoint without `noalias` or an
  equivalent proof. Read/read operations need no edge; writes, volatile and
  atomic operations, fences, and potentially aliasing accesses receive the
  necessary edges. Runtime alias checks may split fast and fallback paths.
  Output is a minimal explicit dependence graph. Below this point there is no
  implicit AA, barrier inference, or loop dependence rediscovery.
- Barriers: `inter.barrier` joins incoming tokens; lowering is `sync.bar`
  plus the required fence sends, with fence scope from the memory-model
  attributes on the op.
- Tiling, unrolling, contraction recovery, and software pipelining must
  transform the dependence graph explicitly. A pass may not discard ordering
  and ask physical instruction analysis to reconstruct source semantics later.

## 10. The `xemachine` dialect

Machine-level MLIR dialect; live from selection through emission. No LLVM
codegen involvement.

### Types

- Above the machine dialect, matrix fragments carry logical shape, element and
  accumulator type, signedness/precision, A/B/C/D role, subgroup/lane
  distribution, VNNI/transpose packing, and target layout. Reuse upstream
  XeGPU layout attributes and Xe2 capability tables where possible; do not
  route through XeVM/OpenCL because Inter owns physical lowering and SWSB.
- Fragment lowering produces ordinary `!xemachine.reg` storage plus explicit
  relative-placement/alignment constraints. Logical tile identity does not
  survive as a recursive machine tuple that hides GRF footprints.

- `!xemachine.reg<width, index>` — GRF storage. `width` is in 32-bit dwords;
  `index` = physical base GRF, `-1` = virtual. Virtual and
  physical share the type; regalloc rewrites the index slot. (Same trick as
  wave-mlir's `!waveamdmachine.reg<class,width,index>`. One bank, so no
  class field; element type lives on the instruction, matching EU encoding.)
- `!xemachine.arf<file, width, index>` — `a0`, `f`, `acc`, `mme`, specials.
  Separate type because ARF allocation is a separate mini-problem with its
  own rules.
- `!xemachine.imm` — SSA immediates (EU allows one immediate source; the
  constraint is checked by an op interface, not by the type alone).
- `!xemachine.mem.token`, `!xemachine.barrier`.

### Ops

- One TableGen base class injecting the interface set into every instruction
  op, metadata spliced from TableGen string fields — the wave-mlir
  `WaveAMDMachine_Op` pattern.
- ALU ops carry: exec size, predication (flag + inverse), flag modifier,
  source regions, destination region, saturation — as attributes with
  defaults, so the common case stays terse.
- Structured control flow as region ops implementing
  `RegionBranchOpInterface`/`LoopLikeOpInterface`:
  `exec_if` (mask-stack push/pop region), `uniform_if`, `uniform_loop` with
  `continue_if` back-edge terminator. The emitter requires one-block funcs;
  all control flow exists only as these region ops at emission time.
- Send ops: descriptor fields as attributes + payload operands; an op
  interface exposes the descriptor spec so the address planner can query
  operand shapes without hardcoding message tables in the planner.
- Named semantic machine sends cover A64 gather/scatter, block/2D
  load/store/prefetch, SLM, fences, and barriers. Target encoding code owns raw
  descriptor bitfields. Selection and tests do not copy descriptor constants.
- `xemachine.dpas` is a first-class instruction op, not a raw ALU escape hatch.
  It records execution size, systolic depth, repeat count, source precisions,
  accumulator/destination precision, fragment footprints, packing, target
  availability, and destructive accumulator/result storage. Its verifier owns
  all shape, alignment, overlap, and contiguous-bundle constraints.
- `tuple_from_elements`, `tuple_to_elements`, and `update_tuple` are zero-cost
  storage views, not recursive tuple types. They expose weighted dword offsets
  through `RegisterStorageAliasOpInterface`; destructive updates are marked
  explicitly. A64 SIMD32 addresses are one 64-dword tuple, so all four payload
  GRFs remain allocator-visible.
- Pseudo ops: `token`, `token_join`, `after`, `reg_after`, copies, payload
  materialization. Pseudos are erased or materialized by emission; they never
  reach the encoder.

### Interfaces (initial set)

- `SWSBInfoOpInterface` — pipe, in/out dependency classes, whether the op
  can hold a token, sync-function eligibility. TableGen-declared per op.
- `RegionLegalityOpInterface` — operand region/alignment constraints.
- `SendDescriptorOpInterface` — descriptor field layout and payload rules.
- `FixedPhysicalRegisterDefsOpInterface` — r0 payload, implicit ARF uses.
- `InstructionIssueOpInterface` — issue latency/throughput classes for the
  cost model.
- `RegisterStorageAliasOpInterface` — arbitrary `Value` storage aliases with
  weighted dword offsets and destructive-use markers. Tuple ops provide local
  edges; region flow adds yields and loop carries to the same alias sets.

Arch gating is a static C++ predicate per op (`isSupportedOn(isa)`), queried
before instantiation. Only `xe2` is populated; the enum leaves room.

Initial DPAS coverage is capability-table driven: f16/bf16 to f32 and selected
i8 signedness combinations. Unsupported type/shape/fast-math combinations stay
as generic contraction loops; they are not rounded into a convenient DPAS form.

## 11. Contraction recovery, layout planning, and selection

Optimized LLVM first reaches the complete semantic SPMD layer. Direct
LLVM-to-machine pattern matching is forbidden.

- General legalization covers exact scalar/fixed-vector arithmetic, casts,
  aggregates, shuffles, masks, branches, and loops. Type expansion/promotion
  preserves overflow, poison, and fast-math behavior. Unsupported semantics
  fail at the declared boundary; they do not fall through a dispatch loop.
- Contraction recovery recognizes semantics rather than syntax: one accumulator
  recurrence, multiply-add/dot-product update, two input addresses sharing a
  reduction dimension, output coordinates independent of that dimension, a
  proved extent/tail, and no interfering dependence. Accepted forms include
  scalar fmul+fadd, explicit FMA, vector FMA plus reduction, masked/VP forms,
  unrolled trees, and main-loop plus epilogue. Near misses remain generic.
- Strict FP is never reassociated into DPAS. FMA contraction requires explicit
  fused semantics or `contract`; reduction reordering requires `reassoc` or an
  equivalent contract. Reduced precision/TF32 requires explicit permission.
- Exact Intel subgroup matrix builtins are decoded through a signature and
  attribute registry, not symbol-prefix matching. Intrinsic and idiom paths
  normalize to the same target-independent contraction operation; intrinsic
  identity does not bypass memory-layout legality.
- Tile planning chooses kernel SIMD width, workgroup/subgroup M/N/K tiles,
  direct-global versus SLM staging, prefetch depth, fragment packing, tails,
  and 128/256-GRF mode from target capabilities, occupancy, SLM, and pressure.
  Dimensions such as 8/16/32 originate only in capability tables.
- Every contraction has a generic ALU/reduction lowering. DPAS is selected only
  when type, fast-math, shape, layout, packing, alignment, and target constraints
  are all proved. Every memory access follows the 2D -> block -> gather/scatter
  fallback chain from section 8.
- Selection emits structured optimization remarks for contraction recovery,
  rejected legality predicates, chosen message fallback, tile/layout choice,
  GRF mode, expected occupancy, and spill estimate. Tests may assert decisions
  through remarks but correctness never depends on their text.
- Subgroup ops lower to regioned moves only after the lane layout is explicit:
  broadcast = `<0;1,0>` region read, shuffle = legal indirect/region sequence,
  reductions = target-planned trees or contraction fragments.
- Control flow maps semantic uniform/divergent regions and loops to structured
  machine operations with explicit carried state and active masks. Uniform and
  divergent loops, continue/break, and edge masks must be emitted end-to-end.
- Threading-model intrinsics lower through the kernel ABI descriptor and target
  payload table. X/Y/Z IDs, group/local sizes, subgroup IDs, and required
  workgroup/subgroup sizes are semantic operations, not hardcoded r0 offsets in
  the generic selector.

## 12. Register allocation

Transform-loop linear scan, transplanted from wave-mlir
`lib/Dialect/Wave/Transforms/RegAlloc/`:

- IR-resident epoch state lives in a function attribute while an allocation
  attempt is active. No C++ state survives an IR rewrite; each relief rewrite
  clears the attribute and the next iteration rebuilds positions, value IDs,
  intervals, weighted alias sets, and send-source lifetimes from current IR.
- Alias preparation runs before scheduling and defensively before every state
  rebuild. It materializes legal parallel copies for destructive updates,
  branch joins, loop entries/backedges, and incompatible relative placements.
  DPAS accumulator chains and send/fragment bundles participate in the same
  storage-constraint graph.
- Scheduler and allocator consume one shared immutable weighted-alias analysis.
  Raw tuple and structured-region constraints must be normalized before this
  analysis; inconsistent weighted cycles remain hard errors after preparation.
- Production allocation uses dword/byte-granular footprints, not the current
  whole-GRF-only prototype. It supports sub-GRF live ranges, per-operand
  alignment, contiguous multi-GRF send/DPAS bundles, relative placement,
  VNNI-packed fragments, destructive accumulator/result ties, fixed payload
  GRFs, and overlapping source/destination regions.
- Linear scan consumes alias state and either commits physical indices or
  emits one precise failure record. It never picks a relief strategy.
- Relief starts with rematerialization and live-range splitting, then scratch
  spill; fragment and loop-carried accumulator spills receive prohibitive cost
  unless no legal allocation exists. Spill/fill supports arbitrary legal
  footprints and rebuilds scheduling/SWSB state. SLM spilling is optional only
  where capacity and synchronization semantics prove it safe.
- Region-aware liveness uses exact `<V;W,H>` byte footprints. ARF allocation
  for flags, `a0`, accumulators, and MME state is a separate constrained
  problem with hard capacity and interference; production selection may not
  require every ARF to be preassigned.
- Pressure accounting: the GRF budget is a function input, not a pass option
  or allocator constant. `xemachine.grf_count` supplies the selected GRF mode
  and `xemachine.reserved_grf_count` supplies the ABI-reserved prefix. The
  budget and occupancy target are the same knob, so selection records the
  per-kernel decision and regalloc only consumes it.
- Every allocation attempt ends with a physical-footprint and instruction-
  constraint verifier. The verifier checks all send and DPAS bundles,
  accumulator ties, region bounds, alignment, fixed-register overlap, and GRF
  mode before SWSB is allowed to run.

## 13. Scheduling

"Scheduler is a stall filler; the cost model owns policy." Copied as law.

Matmul has two distinct scheduling levels:

- Semantic loop scheduling chooses prefetch placement, direct-global versus SLM
  staging, K-loop software-pipeline stages, and double/multi-buffered fragment
  rotation before physical instruction selection. Async issue and await points
  are explicit IR operations; stage count is cost-model data, not kernel code.
- Machine scheduling runs pressure-aware before RA, then a restricted repair
  schedule runs after spill/copy insertion. The final instruction order is
  frozen before SWSB. No instruction moves after scoreboard annotation.

- The baseline machine scheduler walks each machine region in original program order; if the
  next op issues stall-free per the timing oracle, it stays; otherwise a
  legal ready instruction fills the gap. Remaining stalls are left for
  SWSB/sync to make correct.
- The scheduler owns only the dependence graph (SSA edges, token edges,
  singleton edges for flags/address registers, loop carries) and the legal
  ready frontier. All target policy — latencies, pipes, issue rules, filler
  compatibility — lives behind a provider-function interface in the cost
  model.
- `MachineScheduler.*` contains only region collection, dependence mechanics,
  deterministic ready-frontier selection, and IR movement. `Xe2ScheduleModel`
  adapts the Xe2 timing oracle and owns pipes, occupancy, ARF resources, and
  filler compatibility.
- Region collection completes before mutation, splitting at nested structured
  operations and scheduling their blocks recursively. One immutable Xe2 model
  serves the function; each collected region gets fresh issue and pressure
  state.
- The Xe2 model builds the shared function-wide register-alias analysis once.
  Candidate pressure uses exact live byte/dword footprints while preserving
  alias-component placement constraints, fixed GRFs, cross-region live ranges,
  fragment bundles, and send payload lifetime through source-read completion; a
  filler cannot raise the original schedule's peak pressure.
- Ready zero-byte operations are drained to closure before each greedy choice.
  They forward value/token readiness without consuming an issue slot.
- The scheduler never queries memory effects or alias information. Memory
  ordering edges are exactly the explicit token SSA edges synthesized before
  machine selection; unrelated sends remain reorderable.
- One timing oracle shared by scheduler preview and any simulation. No
  second oracle (documented failure mode in wave-mlir).
- Completion latency, issue occupancy, send-source read time, and dependency
  gap rules are adapted from IGC's pinned target scheduler model. XeMachine ops
  classify themselves; the target timing oracle owns the policy constants.
  The scheduler algorithm remains wave-mlir's deterministic greedy gap filler,
  not IGC's list scheduler. The target model additionally owns DPAS completion,
  occupancy/read windows, systolic-pipe resources, block-message timing, SBID
  pressure, and read-suppression legality. SLM-bank/cache effects may influence
  cost but never correctness.

## 14. SWSB annotation

Runs after the final register allocation, spill/copy repair, and instruction
order. RA-inserted operations create dependencies too; no instruction moves
after this pass.

- Dense forward dataflow tracks exact physical GRF/ARF read/write footprints,
  per-pipe producers, send source-read retirement, send destination completion,
  DPAS read/accumulator windows, and region/loop state.
- The pass allocates and reuses hardware SBIDs `0..31`. It frees a token only
  after all represented source/destination obligations complete, propagates
  state across branches/joins/backedges, and inserts `.src`, `.dst`, distance,
  or `sync` dependencies. More than 32 sends is a normal stress case, not an
  encoder failure. If CFG merge or token pressure cannot be represented, drain
  conservatively and continue.
- DPAS dependencies include the systolic pipeline, destructive accumulator
  chain, source-read window, read suppression, and interleaved send completion.
- Output is final SWSB attributes on physical instructions. A verifier then
  simulates every RAW/WAR/WAW, fixed-register, send, DPAS, and loop/join hazard.
  Fault-injection tests must prove that removing any required dependency is
  diagnosed.
- The encoder serializes final attributes and performs no token allocation,
  distance inference, or independent send numbering. Raw SWSB overrides are
  test-only.

`WaveAMDMachineWaitcnt.cpp`'s `WaitLattice` is the structural template;
Intel's counter-less token model replaces the AMD counter model. Mesa elk's
SWSB scheduler is the correctness reference for the rules themselves.

## 15. Emission

A translation (`inter-translate --xemachine-to-zebin`), not a pass. There is
no LLVM MC target for Xe; the MC-equivalent layer is owned here.

- **Encoder: link GED** (`visa/iga/GEDLibrary`, MIT, C API). Machine ops map
  to GED encode calls per op; field values come from op attributes. This is
  the single decision that removes the largest risk in the project — bit
  encoding is Intel's own table, not our transcription. Decoder side gives
  free disassembly for tests and debugging.
- The standalone build fetches IGC v2.38.2 at a pinned commit and archive
  hash. `inter-translate --xemachine-to-ged` emits native 16-byte Xe2
  instructions directly; `--xemachine-to-asm` formats those final encoded
  bytes with IGA's GED decoder, and `--xemachine-to-zebin` packages them as a
  runnable ELF device binary.
- Buffered emission: instructions accumulate in a buffer (variant of
  instruction / label / alignment / directive) so fixups apply before
  finalize: branch targets (JIP/UIP) and optional compaction later. If a layout
  change would invalidate final SWSB, it must happen before SWSB or trigger a
  full re-run; emission never repairs dependencies.
- Selection records the kernel ABI and fixed target-resource inputs as function
  attrs (`kernel_type`, `grf_count`, `reserved_grf_count`, `simd_size`, payload
  sizes, and `slm_size`); regalloc records `scratch_size`. After synchronization,
  `inter-resource-info` validates physical allocation and publishes
  `grf_used`, `barrier_count`, `has_global_atomics`, `has_dpas`, and
  `has_no_stateless_write`. The emitter requires this final resource contract,
  cross-checks it through the shared XeMachine resource analyzer, and
  serializes it without owning a second metadata policy.
- ELF + `.ze_info` + `.note.intelgt.compat` writer; ground truth is NEO's decoder
  (`shared/source/device_binary_format/zebin/`). The emitter is wrong when
  NEO rejects it, by definition.

## 16. Host interface and ABI

### Host path: liboffload, not raw L0

All host-side device interaction goes through LLVM's offload library (in this
repo), never through hand-rolled `ze*` calls:

- `olInit` + platform iteration, selecting `OL_PLATFORM_BACKEND_LEVEL_ZERO`.
- `olCreateProgram(Device, Data, Size, &Program)` loads the module. The
  buffer must be an LLVM **OffloadBinary** container
  (`llvm::object::OffloadBinary::write`) with image kind `IMG_Object` and a
  `spirv64` triple arch — the L0 plugin validates exactly these two fields
  (`L0Plugin.cpp`), then passes the payload to
  `zeModuleCreate(ZE_MODULE_FORMAT_NATIVE)` (`L0Program.cpp`). IGC is never
  involved; NEO loads the native binary directly.
- `olGetSymbol(Program, OL_SYMBOL_KIND_KERNEL, name, &K)` +
  `olLaunchKernel(...)`; `olMemAlloc`/`olMemcpy` for buffers. Working usage
  example: `offload/unittests/OffloadAPI/kernel/olLaunchKernel.cpp`.
- `olIsValidBinary` doubles as a cheap "does NEO accept this zebin" smoke
  check without a launch.
- The zebin writer therefore owns only the device binary; the OffloadBinary
  wrapper is a fixed dozen-line preamble at program-load time. No offload
  changes are needed upstream.

### Device ABI: payload and zeinfo

- One IR-resident kernel ABI descriptor, created at import and consumed by both
  prologue generation and zebin emission, owns explicit/implicit arguments,
  offsets, sizes, alignment, address spaces, access modes, inline/indirect
  placement, local-ID payload, entry offsets, required workgroup/subgroup sizes,
  SIMD width, and GRF mode. No pass recomputes placement from argument index.
- Thread payload has dual entries. The software-local-ID prologue copies
  inline arguments from r1 to r4 and loads local IDs into r1-r3; NEO's
  hardware-local-ID path enters at byte 192 with that same register layout
  already established. Argument placement is described by zeinfo
  `payload_arguments` entries (offset/size/kind per argument).
- Implicit arguments (global offset, enqueued local size, printf buffer,
  scratch pointer, sync buffer, ...) are requested explicitly in zeinfo only
  when the kernel uses them.
- Execution environment in zeinfo: SIMD size, SLM size, barrier count, GRF
  count/mode, DPAS use, required workgroup/subgroup size, atomic use, and
  stateless-write use. These are cross-checked against machine IR and resource
  attrs at emission; mismatch is a hard error.
- The runner accepts configurable 1D/2D/3D group and local sizes, typed
  f16/bf16/f32/i8/i32 buffers and scalars, deterministic/random/file inputs,
  dynamic SLM where supported, multiple outputs, floating tolerance, and a CPU
  reference callback. The current fixed 1D `{32,1,1}` uint32 runner is prototype
  debt, not the matmul test ABI.
- Multi-kernel sections and symbols are emitted independently. One-kernel-only
  support may remain during bring-up, but ABI data structures may not assume it.

## 17. Testing

Three tiers, wave-mlir structure:

1. **LIT + FileCheck unit tests** per pass and per emission feature. Machine
   IR is text; CHECK lines capture SSA names with placeholders.
2. **Disassembly goldens.** `inter-translate --xemachine-to-asm` encodes with
   GED and formats the resulting bytes through IGA's GED decoder; FileCheck
   validates the final assembly. This replaces wave-mlir's text-ASM golden
   tier — binary drift review only works if a human can read the diff. The
   field-oriented `inter-ged-dump` tests remain as direct encoder checks.
3. **End-to-end via `inter-runner`**, built on `liboffload` (section 16):
   wraps emitted zebins in the OffloadBinary container, loads them with
   `olCreateProgram`, and launches them through the generic argument ABI. No
   raw `ze*` calls anywhere in the project. Gated on hardware presence
   (`REQUIRES: host-supports-inter-bmg`).

The optimized-LLVM corpus freezes both `opt -O2` and `opt -O3` output for each
source kernel. Tests include scalar, vectorized, unrolled, peeled, masked/VP,
FMA, reduction, dynamic-stride, transpose, aliasing, and tail forms. O2 and O3
must produce identical answers even when only one form is contraction- or
block-message-eligible.

Lighthouse supplies three references, not selector templates:

- Lighthouse's `test/run/pipeline-check.mlir` is the first reproducible matmul
  corpus. Lower it to LLVM IR, freeze pre-opt plus `opt -O2/-O3` IR, and use its
  deterministic inputs/reference for generic-path bring-up.
- Lighthouse's `examples/xegpu/matmul.py` is the Intel performance and layout
  oracle: f16/bf16->f32, M8xN16xK16 DPAS geometry, 2D block
  load/store/prefetch, transpose, bias/ReLU, Level Zero validation, and staged
  MLIR dumps. Inter does not pattern-match its generated names or operation
  order.
- The pinned KernelBench `level1/2_Standard_matrix_multiplication_` becomes the
  production corpus once initialized. Capture its LLVM-dialect output, translate
  to LLVM IR, run `opt -O2/-O3`, and validate against PyTorch before benchmarking.

Differential tests compare generic and optimized paths on dynamic M/N/K,
arbitrary leading dimensions, non-square and irregular shapes, transpose A/B,
tails, aliasing near misses, strict FP versus fast-math, alpha/beta, bias,
batched layouts, and multiple launch geometries. Every optimization has positive
legality tests, structurally perturbed positive cases, and near-miss negatives.

The M11 performance tier uses frozen pipeline inputs plus disassembly goldens;
drift is a review stop with a mandatory A/B benchmark on the B60.

## 18. Milestones

- **M0-M4 — executable backend spine (complete prototype).** Native zebin,
  LLVM import, straight-line integer kernels, branches, SLM/barriers/atomics,
  pre-RA scheduling, whole-GRF allocation, conservative synchronization, GED
  encoding/disassembly, and B60 hardware tests. These prove the container and
  machine path; they do not certify a general LLVM lowering.
- **M5 — optimized LLVM semantic baseline.** Freeze Lighthouse's small matmul
  as pre-opt, O2, and O3 LLVM IR. Introduce explicit conversion legality,
  kernel/helper identification, proof/metadata preservation, exact scalar and
  fixed-vector arithmetic, general reducible loops, masks/tails, and exact
  address objects. Delete `gid*4`, direct-GEP/block-argument, argument-index,
  i32-only, and X-only selection shortcuts. Acceptance: O2 and O3 generic paths
  execute multiple dynamic/irregular f32 matmuls correctly with no block message
  or DPAS required; unknown LLVM operations fail loudly.
- **M6 — general ABI and execution model.** Implement the shared ABI descriptor,
  SIMD8/16/32 selection, X/Y/Z and subgroup/workgroup builtins, configurable
  1D/2D/3D launch, typed buffers/scalars, arbitrary argument payload coverage,
  and CPU-reference validation. Acceptance: the same generic matmul runs at
  multiple shapes, leading dimensions, launch geometries, and SIMD widths.
- **M7 — semantic contraction and layout level.** Recover legal contractions
  from scalar/vector/FMA/reduction O2/O3 forms and normalize exact Intel matrix
  intrinsics. Add target-independent fragment/layout operations, strict-FP
  gates, dynamic tails, transpose/VNNI conversions, and generic fallback.
  Acceptance: structurally different O2/O3 reductions converge on equivalent
  contraction IR when legal; strict/near-miss cases remain generic and correct.
- **M8 — block/2D memory vertical slice.** Add target capability tables, named
  block/2D load/store/prefetch ops, descriptor/payload construction, timing,
  verification, and 2D->block->gather fallback. Acceptance: eligible A/B/C tiles
  from the Lighthouse XeGPU reference disassemble as block2D messages; dynamic
  misalignment, pitch, transpose, and edge cases take tested fallbacks.
- **M9 — DPAS vertical slice.** Add `xemachine.dpas`, fragment footprints and
  accumulator aliases, f16/bf16->f32 precision support, GED setters, timing and
  issue resources, `has_dpas` resource/zeinfo metadata, and IGA goldens.
  Acceptance: a hand-authored then selected M8xN16xK16 subgroup tile encodes the
  expected DPAS and matches the generic path on B60.
- **M10 — production physical pipeline.** Add sub-GRF constrained allocation,
  contiguous fragment/send bundles, ARF/MME allocation, 128/256-GRF cost choice,
  fragment-aware spill/splitting, post-RA scheduling, reusable SBID allocation,
  DPAS/send SWSB, and final hazard verification. Acceptance: loops with more
  than 32 sends, double-buffered fragments, spills, branches, and backedges pass
  fault-injected verifier and hardware stress tests.
- **M11 — production matmul qualification.** Software-pipeline the K loop,
  overlap block prefetch/load with DPAS, choose direct-global versus SLM staging,
  and tune through target cost data. Validate Lighthouse XeGPU f16/bf16 kernels
  and KernelBench level1/2 O2/O3 IR across dynamic/non-square/tail/transpose/
  bias/ReLU cases before adding performance gates. Acceptance: correctness,
  NEO zebin validation, reviewed final assembly, stable resource usage, and a
  documented B60 performance baseline against the generic Inter path.

## 19. Risks and open questions

- **SWSB debuggability.** Failures are silent corruption. Mitigation:
  start over-conservative, require the independent physical-hazard verifier,
  fault-inject missing dependencies, stress token reuse across loops/joins, and
  keep disassembly goldens.
- **Optimized-LLVM shape drift.** LLVM releases and `-O2/-O3` produce different
  loop, vector, and intrinsic forms. Mitigation: freeze representative IR from
  each supported toolchain, test semantics rather than exact operation order,
  and retain the generic path whenever contraction recovery does not fire.
- **False contraction recovery.** A syntactically plausible reduction may have
  aliasing, strict-FP, poison, overflow, or loop-carried side effects that make
  reassociation invalid. Mitigation: use imported LLVM proofs and explicit
  legality predicates, compare optimized and generic paths, and maintain more
  negative/near-miss tests than positive patterns.
- **Fragment-layout completeness.** DPAS operand packing, VNNI, transpose,
  block-message result layout, and accumulator layout must agree exactly.
  Mitigation: one target capability/layout table feeds selection, verifiers,
  allocation constraints, GED emission, and IGA assembly goldens; no pass keeps
  an independent shape table.
- **2D message dynamic legality.** Runtime base alignment, pitch, dimensions,
  and edge behavior may invalidate an otherwise useful tile. Mitigation: prove
  static legality, guard profitable dynamic cases, and always retain block or
  masked gather/scatter fallback.
- **Region legality completeness.** The compatibility matrix lives in GED
  validation and elk source, not in any document. Expect a legality-test
  sweep enumerating producer/consumer region pairs against GED's validator.
- **Errata.** ISA-level workarounds are not published as a list; they are
  visible in IGC/NEO source. Budget for source archaeology when hangs
  appear.
- **Payload-format drift.** The thread-payload ground truth is the DG2 PRM,
  one generation behind the target with no public Xe2 PRM. Wrong argument
  offsets fail silently, same failure class as SWSB. Cross-check the payload
  layout against elk's compute payload code and validate at M0 with a kernel
  that reads every argument and writes it back.
- **GED version skew.** Pin the IGC commit; Xe2 tables are stable but the
  API is not frozen.
- **zebin drift.** Pin the NEO commit used for decoder ground truth; CI
  should fail loudly on decoder rejection, not on our writer's opinion.
- **Latency data.** Intel does not publish a B60 latency table. Use the timing
  constants and heuristics in the pinned IGC scheduler as the source of truth,
  preserving latency, occupancy, and send-source read time as distinct values.
- **Exec-width strategy.** Compile each kernel variant at one width chosen
  before semantic layout planning (normally SIMD16 for the initial DPAS path;
  SIMD8/32 where legal and profitable). Future multiversioning creates separate
  variants and zeinfo entries rather than making width dynamic inside a kernel.
- **Performance cliffs.** DPAS selection can lose to generic code through poor
  occupancy, spills, excess layout conversion, or tail overhead. Mitigation:
  keep block messages and DPAS as independent choices, cost 128/256-GRF modes,
  report selection remarks/resource usage, and require B60 A/B data before
  making an optimization default.

## 20. Transplant map from wave-mlir

All paths below are relative to the wave-mlir repo (`github.com/Hardcode84/7`)
at the pinned commit from section 3.

| inter component | wave-mlir source | Verdict |
|---|---|---|
| Machine dialect TableGen shape | `include/mlir/Dialect/WaveAMDMachine/IR/*.td` | direct, minus AMD encodings |
| Token types + threading | `WaveOps.td` token ops, `WaveLowerTokenSelects.cpp` | direct |
| Token synthesis (AA) | — | new; wave-mlir refuses this by policy |
| Symbolic engine | `third_party/ixsimpl`, `WaveSymbols.*`, `WaveGenerateIndexExprs.cpp` | direct; submodule |
| Address planner shape | `WaveAMDMachineIndexExpr.cpp`, `WaveAMDMachineSelector.h` | shape direct; slot rules are Intel message trivia |
| Uniformity and lane distribution | bare uniform values plus `!wave.simd<T, W>` | bare/SIMD distinction is direct; compact `N < W` and affine proofs are Inter extensions, with `WaitLattice`/`HazardLattice` as dataflow exemplars |
| Structured machine CF | `uniform_loop`/`uniform_if`/`exec_if`, `WaveAMDMachineScfFor.cpp`, `WaveAMDExecIfUtils.cpp` | direct concept match to EU CF + mask stack |
| Regalloc transform loop | `lib/Dialect/Wave/Transforms/RegAlloc/` | direct in shape; drop AGPR provider; region aliasing is new |
| Scheduler + cost model split | `WaveAMDMachineGreedySchedule.cpp`, `WaveAMDMachineScheduleModel.*`, `CostModel/` | greedy scheduler architecture direct; timing policy adapted from IGC |
| SWSB pass | `WaveAMDMachineWaitcnt.cpp` (`WaitLattice`, physical source tickets) | closest analog; counter model replaced |
| Resource info / metadata attrs | `WaveAMDResourceInfo.cpp`, `WaveAMDMetadata.cpp` | pattern direct; zeinfo replaces HSA metadata |
| Emission buffering + fixups | `lib/Target/Wave/AMDGPU.cpp` buffered MC pattern | pattern direct; GED replaces MCInst/printer |
| Pipeline-as-data | `lib/Target/Wave/pipelines/pipelines.mlir` | direct |
| Test tiers | `test/lit.cfg.py`, `test/PerfGolden/`, `test/Integration/` | direct, with GED-disasm goldens replacing text goldens |
| LLVM IR front | — | new; upstream `import-llvm` + our cleanup |

Design disciplines carried over verbatim: verifiers stay local; no string
round-trips across FFI; no hidden C++ state across IR rewrites; scheduler
builds legal ready sets and the cost model owns policy; one timing oracle;
regalloc reports and providers decide.

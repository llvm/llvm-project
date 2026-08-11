# inter: Intel Xe2 GPU Backend in MLIR — Design

Status: draft. Project root: `inter/` in this repo. First target: Xe2 / Battlemage
G21 (Arc Pro B60, PCI 8086:e211), Linux, `xe` KMD, Level Zero runtime.

## 1. Goal

A compiler:

```
LLVM IR (spir64 kernel shape) -> MLIR LLVM dialect -> scf -> xemachine dialect -> Intel EU binary -> zebin ELF -> Level Zero
```

No IGC, no vISA, no SPIR-V, no NEO compiler interfaces on the device path.
The host side uses LLVM's `liboffload` (in this repo) over Level Zero as
module loader and runtime; the only L0 entry point ever exercised is
`zeModuleCreate(ZE_MODULE_FORMAT_NATIVE)`. All device-side knowledge below
the machine dialect — encoding, scoreboarding, container format — is owned
here.

Correctness before performance. Every pipeline stage is printable, inspectable
MLIR. No pass keeps hidden C++ state across an IR rewrite.

## 2. Non-goals (v1)

- XMX / `dpas` and matrix fragment layouts.
- Instruction compaction (64-bit forms). v1 emits 128-bit instructions only.
- Function calls, relocations, indirect branches, SIP/debug support.
- Ray tracing, media fixed functions, bindless-only kernels.
- Large GRF mode (256 GRFs/thread). v1 targets the 128-GRF mode only; see
  sections 4 and 12.
- Multiple Intel platforms. Everything is Xe2; arch gating exists but only
  `xe2` is populated.
- Performance. The scheduler starts as a legal-order passthrough with a stub
  cost model.

## 3. Ground truths

The ISA and container are fully determined by open sources. Nothing here
requires reverse engineering.

| What | Where | Role |
|---|---|---|
| EU encoding, Xe2 bit-exact | `intel/intel-graphics-compiler`, `visa/iga/GEDLibrary` (GED) | encoder/decoder library, MIT, C API; XE2 tables annotated per-field in `ged_enumerations.h` |
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
  couples the register budget to occupancy; v1 compiles for 128-GRF mode
  only (section 12), but nothing in the dialect assumes 128. Plus small ARF
  files: address registers `a0`, flags `f0/f1`, accumulators `acc`, `mme`,
  and specials (`sr0`, `cr0`, `n0`, `ip`, `tm0`, ...). No separate scalar
  bank — uniformity pays through message selection and region narrowing, not
  through a cheaper register class.
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
LLVM IR (spir64)
  | mlir-translate --import-llvm                     (upstream)
  v
llvm dialect + func
  | inter-import-cleanup: canonicalize, strip host IR, normalize
  |   kernel metadata, attach xemachine.target attr
  v
llvm/cf mix
  | lift-cf-to-scf                                    (upstream; rejects
  |                                                     irreducible CFGs)
  v
scf + arith + llvm remnants
  | inter-uniformity-analysis      (dense dataflow; section 7)
  | inter-generate-index-exprs     (SSA -> ixsimpl; section 8)
  | inter-memory-token-synthesis   (early AA -> explicit tokens; section 9)
  v
annotated scf level  === selection boundary ===
  | inter-select-to-machine
  v
xemachine (virtual regs, tokens, region CF ops)
  | machine opts: copy folding, region narrowing, message-form selection
  |   via symbolic address planner, dead code, canonicalization
  v
  | inter-machine-schedule         (stall filler + cost model; section 13.
  |                                 Scheduling is pre-RA by design: the scan
  |                                 needs freedom to move virtual-reg code;
  |                                 RA copies get their dependencies from the
  |                                 post-RA SWSB pass, section 14)
  v
  | inter-regalloc transform loop  (section 12)
  v
xemachine (physical regs)
  | inter-insert-sync              (conservative explicit waits; section 14)
  | inter-resource-info            (GRF/SLM/spill/barrier counts -> attrs)
  v
emission (translation, not a pass)
  | GED encode, buffered fixups, branch targets
  | zeinfo YAML + ELF write
  v
zebin ELF  ->  OffloadBinary wrapper  ->  liboffload olCreateProgram
           ->  zeModuleCreate(ZE_MODULE_FORMAT_NATIVE)
```

Pipeline is data: a `transform.with_named_sequence` library interpreted by
`transform-interpreter`, same as wave-mlir's `pipelines.mlir`. All dialects
are preloaded before the interpreter runs (its multi-threaded context refuses
late registration — known pitfall, documented in wave-mlir).

Stage discipline: one dialect mix per stage, every boundary FileCheck-able,
tests enter at named entry points (`@inter_backend_preschedule`,
`@inter_backend_emit_only`, ...).

## 6. Frontend: LLVM IR import

- Input contract: clang `spir64` output. Address spaces: private=0, global=1,
  constant=2, local=3, generic=4. `spir_kernel` calling convention. SPIR-V
  intrinsics for IDs/barriers/subgroups.
- Upstream `import-llvm` produces the llvm dialect. A cleanup pass converts
  what remains to arith/func form and rejects unsupported shapes with hard
  errors (no silent feature drops).
- Irreducible CFGs are rejected at `lift-cf-to-scf`. Kernels from structured
  frontends are always reducible; if irreducible input ever matters, add a
  structurizer pass rather than weakening the machine dialect.
- The function-signature -> payload-argument mapping is computed here and
  carried as attributes; zeinfo emission later just serializes it.

## 7. Uniformity analysis

Dense forward dataflow over the MLIR dataflow framework, above selection.

Lattice per value (join in this order):

```
const  <  uniform  <  affine-strided(k, base-uniform)  <  varying
```

- `const`: compile-time known.
- `uniform`: same in all lanes of the subgroup (and, where provable, the
  workgroup — needed for barrier-safety reasoning).
- `affine-strided`: lane-affine with uniform base and constant stride k.
  This is the class that selects block vs. scatter message forms; it is why
  boolean uniformity is insufficient.
- Sources: lane id (strided k=1), workgroup id (uniform), constants, and
  propagation rules through arith/scf ops. Region-boundary propagation via
  `RegionBranchOpInterface`.
- The strided-ness proofs delegate to ixsimpl range/divisibility queries
  under assumptions (`wave.assume`-style: predicates attach to SSA results,
  recovered only through def chains).
- Results are consumed by: message-form selection (block/gather/scatter),
  branch classification (uniform condition -> `uniform_if`/`uniform_loop`,
  else `exec_if`), and region narrowing (uniform producer allows
  `<0;1,0>`-style broadcast reads).
- Analysis results are advisory annotations. Nothing below selection may
  re-derive uniformity; passes read the annotation or the explicit type.

Exemplars to mine for lattice plumbing: `WaitLattice`/`HazardLattice` in
wave-mlir (`lib/Dialect/Wave/Transforms/WaveAMDMachineWaitcnt.cpp`,
`WaveAMDHazardWaits.cpp`) — dense dataflow over region-bearing control flow,
join/back-edge handling included.

## 8. Symbolic addressing

Reuse `ixsimpl` (C99, hash-consed, arena DAG; wave-mlir repo,
`third_party/ixsimpl` at the pinned commit)
as a submodule. Same ownership rule as wave-mlir: algebra lives in ixsimpl,
SSA and target policy live in the dialect; passes never walk expressions to
prove equality.

- Carrier op `inter.index_expr`: symbolic expr attribute + named SSA
  bindings. Structural identity gives free CSE.
- `inter-generate-index-exprs` reconstructs expressions from arith SSA chains
  (depth-capped walk with `IntegerRangeAnalysis` under a `DataFlowSolver`).
  Frontends that already know index structure may emit `index_expr` directly.
- The address planner (at selection time) classifies each address as
  `{ const-slot, uniform-base, lane-affine, full-remainder }` against the
  target message's operand spec, proves what fits via range queries, and
  demotes along a fixed chain to a general A64 gather/scatter form.
  Message forms v1: stateless A64 byte/dword scatter and A64 block load/store.
  2D block messages are post-v1.
- Address-space lowering: global -> A64 stateless UGM sends; local -> SLM
  sends; constant -> stateless read-only; private -> GRF when promotable,
  else scratch surface via the per-thread scratch pointer implicit argument.

## 9. Memory model: tokens and early alias analysis

Memory ordering is explicit. Legality is SSA dominance plus token edges. No
pass below token synthesis may infer ordering.

- `!inter.mem.token`; memory ops take a dependency token operand and return
  a result token. `join` merges, `after` sequences without data dependence.
- Two tiers: completion tokens (data ready) and issue tokens (ordered issue,
  no completion promise). Sends consume issue-order edges; their completion
  is modeled by SWSB, not by the token graph — the token graph orders
  *issue*, the scoreboard orders *writeback*.
- Tokens thread through regions as ordinary SSA: yielded by `scf.if` /
  machine `exec_if`, carried by `scf.for` / `uniform_loop` iter args.
- **Token synthesis is the one place alias analysis exists.** It runs at the
  scf level, above selection, where ixsimpl can prove non-overlap of
  lane-affine addresses and kernel-argument provenance (`noalias`, distinct
  buffer arguments) is still visible. Output is a minimal explicit token
  graph. Below this point the wave-mlir religion applies: no implicit AA, no
  barrier inference, no loop-carried memory dependencies rediscovered by
  transforms.
- Barriers: `inter.barrier` joins incoming tokens; lowering is `sync.bar`
  plus the required fence sends, with fence scope from the memory-model
  attributes on the op.

## 10. The `xemachine` dialect

Machine-level MLIR dialect; live from selection through emission. No LLVM
codegen involvement.

### Types

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

## 11. Selection

`scf` + arith + llvm-remnants -> `xemachine` with virtual registers.

- Type legalization here: i64 mul/div expansion (no integer divide
  instruction; sequences lifted from Mesa elk), i8/i16 ALU promotion,
  vector shreds to per-lane form.
- Subgroup ops lower to regioned moves: broadcast = `<0;1,0>` region read,
  shuffle = address-register indirect move, reductions = strided region ops.
- Control flow maps one-to-one: `scf.if` -> `exec_if`/`uniform_if` (via
  uniformity annotation), `scf.for`/`scf.while` -> `uniform_loop`/
  predicated loop forms.
- Threading-model intrinsics (`local_id`, `workgroup_id`, ...) lower to
  payload reads per the PRM payload layout (mirrored from elk's compute
  payload code).

## 12. Register allocation

Transform-loop linear scan, transplanted from wave-mlir
`lib/Dialect/Wave/Transforms/RegAlloc/`:

- IR-resident epoch state lives in a function attribute while an allocation
  attempt is active. No C++ state survives an IR rewrite; each relief rewrite
  clears the attribute and the next iteration rebuilds positions, value IDs,
  intervals, weighted alias sets, and send-source lifetimes from current IR.
- Linear scan consumes alias state and either commits physical indices or
  emits one precise failure record. It never picks a relief strategy.
- The v1 relief provider chain is `Remat -> Scratch spill`; the first legal
  plan wins. SLM spilling and the AMD-specific register-file providers are not
  present. Scratch spill/fill uses Xe2 LSC UGM transpose messages and a
  register extended descriptor in `a0.2`. Relief never splits an alias
  component; v1 providers currently accept singleton components only.
- New work vs. the AMD original: region-aware aliasing (`<V;W,H>` overlap
  between a def and its uses constrains placement; 64-bit alignment; exec
  width affects footprint). ARF allocation for `a0`/`f` remains a separate
  predecessor problem with hard capacity; the GRF pass rejects virtual ARFs
  rather than folding them into GRF pressure.
- Pressure accounting: the GRF budget is a function input, not a pass option
  or allocator constant. `xemachine.grf_count` supplies the selected GRF mode
  and `xemachine.reserved_grf_count` supplies the ABI-reserved prefix. The
  budget and occupancy target are the same knob, so selection records the
  per-kernel decision and regalloc only consumes it.

## 13. Scheduling

"Scheduler is a stall filler; the cost model owns policy." Copied as law.

- Walks each straight-line machine region in original program order; if the
  next op issues stall-free per the timing oracle, it stays; otherwise a
  legal ready instruction fills the gap. Remaining stalls are left for
  SWSB/sync to make correct.
- The scheduler owns only the dependence graph (SSA edges, token edges,
  singleton edges for flags/address registers, loop carries) and the legal
  ready frontier. All target policy — latencies, pipes, issue rules, filler
  compatibility — lives behind a provider-function interface in the cost
  model.
- One timing oracle shared by scheduler preview and any simulation. No
  second oracle (documented failure mode in wave-mlir).
- Latency data ships as a JSON overlay produced by a microbenchmark harness
  on the B60; compiled-in defaults are placeholders. Nothing perf-critical
  is compiled in.

## 14. SWSB annotation

Runs after regalloc (RA-inserted copies create dependencies too — same
ordering lesson as wave-mlir's ticket waits after regalloc).

- Dense forward dataflow over the region CFG, tracking in-flight
  producers per pipe and per register range, plus send-issue state.
- v1 policy is conservative correctness: every producer/consumer hazard gets
  a dependency annotation; sync functions where token tracking cannot
  express the dependency. Token budget per hardware rules is never exceeded;
  the pass prefers correctness-preserving `sync` over clever reuse.
- Join policy at region boundaries: merge in-flight state conservatively
  (union of producers; distances recomputed). Documented in the pass, not
  implied.
- Output lives as attributes on instruction ops (not fixed in place until
  emission, since emission-time fixups like branch-target patching may
  adjust layout). Final encode happens in the emitter's buffered stage.

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
  instructions directly; `--xemachine-to-zebin` packages them as a runnable
  ELF device binary. There is no assembly-text emission path.
- Buffered emission: instructions accumulate in a buffer (variant of
  instruction / label / alignment / directive) so fixups apply before
  finalize: branch targets (JIP/UIP), SWSB finalization if layout moved,
  optional compaction later.
- Selection currently preserves the kernel ABI and fixed target resources as
  function attrs (`kernel_type`, `grf_count`, `reserved_grf_count`,
  `simd_size`, payload sizes, `scratch_size`, and `slm_size`). The emitter
  validates those attrs and derives operation-local
  facts such as barrier, atomic, and stateless-write use. M4 moves all of this
  into the dedicated resource-info pass.
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

- Thread payload has dual entries. The software-local-ID prologue copies
  inline arguments from r1 to r4 and loads local IDs into r1-r3; NEO's
  hardware-local-ID path enters at byte 192 with that same register layout
  already established. Argument placement is described by zeinfo
  `payload_arguments` entries (offset/size/kind per argument).
- Implicit arguments (global offset, enqueued local size, printf buffer,
  scratch pointer, sync buffer, ...) are requested explicitly in zeinfo only
  when the kernel uses them.
- Execution environment in zeinfo: SIMD size, SLM size, barrier count, GRF
  count, atomic use, and stateless-write use. These are cross-checked against
  the machine IR and resource attrs at emission; mismatch is a hard error.
- v1: one kernel per module is supported end-to-end; the writer already
  emits per-kernel sections so multi-kernel is format-trivial later.

## 17. Testing

Three tiers, wave-mlir structure:

1. **LIT + FileCheck unit tests** per pass and per emission feature. Machine
   IR is text; CHECK lines capture SSA names with placeholders.
2. **Disassembly goldens.** Emit binary, decode with GED (or `iga -p xe2
   -d`), FileCheck the disassembly. This replaces wave-mlir's text-ASM
   golden tier — binary drift review only works if a human can read the
   diff, and the decoder gives us that. A small number of binary smoke
   goldens guard the encoder directly.
3. **End-to-end via `inter-runner`**, built on `liboffload` (section 16):
   wraps emitted zebins in the OffloadBinary container, loads them with
   `olCreateProgram`, and launches them through the generic argument ABI. No
   raw `ze*` calls anywhere in the project. Gated on hardware presence
   (`REQUIRES: host-supports-inter-bmg`).

Perf tier (later): frozen pipeline inputs + disassembly goldens for perf
kernels; drift is a review stop with mandatory A/B benchmark on the B60.

## 18. Milestones

- **M0 — container proof.** Hand-written kernel (assembled with IGA), wrapped
  by our zebin writer, loaded and launched on the B60 through `liboffload`
  (section 16). Validates zeinfo/payload contract before any codegen exists.
- **M1 — straight line.** LLVM import -> scf -> selection for arithmetic +
  stateless A64 load/store; GED emission; `a[i] = b[i] + c[i]` end-to-end.
- **M2 — control flow.** `exec_if`/`uniform_if`/`uniform_loop`, predication,
  flags; uniform vs divergent branch selection via the uniformity analysis.
- **M3 — memory model.** SLM, barriers, fences, atomics; token synthesis
  with symbolic AA; message-form selection via the address planner.
- **M4 — regalloc + SWSB.** Transform-loop RA with spills; conservative SWSB
  pass; torture tests (deep register pressure, nested CF, mixed sends).
- **M5 — performance.** Real latency tables from microbench calibration,
  scheduler engagement, block-message selection, compaction, then `dpas`.

## 19. Risks and open questions

- **SWSB debuggability.** Failures are silent corruption. Mitigation:
  over-conservative v1, a verification mode that cross-checks annotated
  dependencies against a def-use recomputation, and disassembly goldens.
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
- **Latency data.** Nothing public. The microbench harness is the plan;
  until it exists, the scheduler is a passthrough and perf claims are void.
- **Open: exec-width strategy.** Compile per-kernel at a single width chosen
  from zeinfo constraints vs. multi-width specialization. v1: single width,
  chosen up front (default SIMD16, SIMD32 where legal and profitable).
- **Open: 2D block messages and `dpas`.** Post-v1; fragment layout machinery
  does not transfer from wave-mlir (DPAS packing != WMMA layouts), only the
  typed-fragment-with-verifier pattern does.

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
| Uniformity analysis | — (declared in types there) | new; `WaitLattice`/`HazardLattice` as dataflow exemplars |
| Structured machine CF | `uniform_loop`/`uniform_if`/`exec_if`, `WaveAMDMachineScfFor.cpp`, `WaveAMDExecIfUtils.cpp` | direct concept match to EU CF + mask stack |
| Regalloc transform loop | `lib/Dialect/Wave/Transforms/RegAlloc/` | direct in shape; drop AGPR provider; region aliasing is new |
| Scheduler + cost model split | `WaveAMDMachineGreedySchedule.cpp`, `WaveAMDMachineScheduleModel.*`, `CostModel/` | architecture direct; all tables replaced |
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

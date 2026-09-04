# Memory Ordering and Physical Synchronization

## Two separate problems

Inter deliberately separates source-level memory ordering from physical Intel
EU readiness:

1. `inter-infer-memory-tokens` runs in XW and creates semantic ordering edges
   using memory effects and alias analysis.
2. `inter-insert-sync` runs after physical register allocation and creates SBID
   assignments, waits, drains, and ALU distance dependencies.

The scheduler sits between these stages. It consumes token SSA for memory order
but does not infer aliasing. Conflating the stages would either lose source
semantics or force SWSB decisions before physical register spans are known.

## Semantic memory tokens

### Representation

XW memory operations consume optional `!xw.mem.token` dependencies and return
tokens. Existing explicit dependencies are retained. `xw.token` creates a root,
and `xw.join` combines multiple dependencies.

### Memory frontier

`MemoryFrontierAnalysis` is a forward dataflow analysis over operations with
default-resource read, write, or free effects. Reachable effects accumulate in
the frontier; alias filtering occurs while dependency plans are built.

For a prior/current pair:

- barrier on either side creates a hazard;
- read/read creates no dependency;
- a pair containing a write or free creates a dependency unless every represented
  location is proven `NoAlias`;
- missing location information conservatively aliases everything.

Duplicate dependencies are removed. One dependency is used directly; multiple
dependencies create `xw.join`.

### Prefetch

Block2D prefetch is issue-only semantic ordering. Its token is deferred until the
next potentially aliasing read. Non-aliasing reads do not consume it. Pending
prefetch state is propagated through supported structured regions and restored
to the enclosing frontier at block exit.

### Structured control

Supported region holders are:

- `scf.if`;
- `xw.where`;
- `scf.for`;
- `scf.while`.

If/where arms receive and yield tokens. Loops gain token iter operands, block
arguments, results, and explicit backedges. Missing else regions are created
when needed to forward the incoming token. Barriers consume the current live
frontier and begin a new chain.

Functions and relevant nested regions must be single-block. Memory operations
inside unsupported region holders fail.

### Current semantic limits

- One token type represents ordering; issue-only distinctions are made during
  machine selection with `xemachine.after`.
- Alias decisions use current MLIR `AliasAnalysis`, not imported MemorySSA,
  TBAA, alias scopes, or runtime checks.
- Volatile, atomic scope/order, and barrier scope are not modeled beyond current
  operation memory effects.
- No explicit semantic kernel-exit join is created; selection connects current
  machine ordering to EOT.

## Machine asynchronous issues

`AsyncScoreboardOpInterface` identifies sends and DPAS. Each issue creates an
obligation containing:

- static issuing operation;
- physical source spans;
- physical destination spans;
- source-read pending state;
- destination-completion pending state.

All asynchronous operations begin with pending source reads. Loads, atomics, and
DPAS also have pending destinations; store-like sends normally do not.

An obligation remains live until a required wait, drain, barrier source drain,
physical conflict, or consumer retires the relevant phase. Physical SWSB analysis
does not assume completion merely because estimated cycles elapsed.

## Physical register spans

GRF spans use physical base times 64 bytes plus declared width. ARFs occupy a
synthetic disjoint address range. Tuple aliases can expose a wider containing
storage span.

Overlap detects:

- RAW: reading a pending destination;
- WAR: overwriting a pending source;
- WAW: overwriting a pending destination;
- send payload reuse before source retirement;
- destination reuse before completion.

SSA issue tickets supplement physical spans so direct consumers remain tied to
the correct asynchronous producer.

ALU distance inference enumerates execution size, regions, subregisters, strides,
and element widths, then rounds touched bytes to hardware GRF dependency buckets.
Distinct subregisters in one GRF therefore share a distance dependency bucket.

## SBID allocation

The available scoreboard IDs depend on GRF mode:

- up to 128 GRFs: 16 IDs;
- above 128 GRFs: 32 IDs.

The pass first computes converged issue state, then creates a deterministic
interference graph.

- Every send is one allocation group.
- Accumulator-connected DPAS instructions form one chain group.
- Issues that may remain pending together interfere.
- Mutually exclusive branch issues may share an ID.
- A static issue keeps one ID across loop re-execution and waits before reuse.
- Legal existing assignments remain pinned.

Coloring chooses the lowest unused ID. If all IDs conflict, it chooses the ID
with the fewest colored conflicts and inserts the required wait before reuse.
Every reachable or unreachable async operation receives a legal assignment.

## Waits and drains

Token modes are:

- `set`: assign the issue's SBID;
- `source`: wait until source reads retire;
- `destination`: wait for destination completion, which also subsumes source
  retirement.

Materialization rules include:

- one wait as `sync.nop` with SBID and mode;
- multiple destination waits as selective `sync.allwr` with a bit mask;
- `sync.allrd` retires source phases;
- unmasked `sync.allwr` retires destination-bearing issues;
- `sync.bar` retires source phases but not destinations;
- EOT, full-drain operations, and payload-prologue exit drain all relevant work.

`xemachine.after` is completion-free: it preserves issue order while deliberately
dropping destination-completion propagation for operations such as prefetch.

## ALU distance dependencies

Integer and floating ALU operations use distance aging. Send and DPAS readiness
uses token dependencies.

- Age begins at one and advances only on the same ALU pipe.
- Ages above seven expire.
- Same-pipe RAW uses that pipe.
- Cross-pipe RAW and WAR/WAW use all-pipe distance.
- Joins retain the minimum conservative age.
- The first direct `a0` write receives floating-pipe distance one.

When an encodable instruction cannot carry every required field, synchronization
is split. In particular, DPAS cannot carry its token assignment and an ALU
distance simultaneously in the current lowering, so a preceding `sync.nop`
holds the distance and DPAS retains the SBID set.

## DPAS chains

DPAS destination destructively aliases its accumulator. Chain predecessors are
exempt from ordinary accumulator-definition and SBID-reuse waits. Consumers
outside the chain wait for destination completion. Every member of one chain
must use the same preassigned SBID.

This grouping handles accumulator continuation. Inter does not currently form
hardware Atomic DPAS macros or independently model source-suppression/read-window
rules.

## Final annotations and emission

The pass writes final `swsbPipe`, `swsbDistance`, `swsbToken`, and
`swsbTokenMode` attributes. `XeMachineLowering.cpp` copies them into the emission
program. For direct use on hand-authored machine IR, lowering also verifies and
supplies the mandatory first direct `a0` floating-pipe distance when absent.

GED emission validates:

- distance at most seven;
- token at most 31;
- every distance names a pipe;
- combined distance/token forms are legal for the instruction;
- floating distance is not combined with a send token.

Emission does not allocate tokens or infer general dependencies. The first-`a0`
rule above is its sole defensive distance insertion.

## Validation and limits

Synchronization validation rejects unavailable SBIDs, invalid selective masks,
inconsistent DPAS-chain assignments, and illegal sync forms. Subsequent resource
analysis rejects virtual or out-of-range physical registers, and GED rejects
unencodable final fields.

There is no independent post-SWSB simulator that reconstructs all physical
hazards and proves the annotations sufficient. Tests exercise the inference
algorithm and hardware integration stresses the emitted result, but fault-
injection completeness is not implemented.

Other current limits include:

- single-block structured semantic token inference;
- no post-SWSB instruction movement;
- no timing-based retirement of async obligations;
- no dedicated DPAS source-suppression model;
- 256-GRF SBID planning exists, while current zebin emission remains 128-GRF.

## Normative sources and tests

- Semantic frontier: `include/inter/Analysis/MemoryFrontierAnalysis.h` and
  `lib/inter/Analysis/MemoryFrontierAnalysis.cpp`
- Token inference: `lib/inter/Transforms/InterInferMemoryTokens.cpp`
- Physical sync: `lib/inter/Transforms/InterInsertSync.cpp`
- Interfaces: `include/inter/Dialect/XeMachine/IR/XeMachineInterfaces.td`
- Encoding: `lib/inter/Emit/GedBinaryEmitter.cpp`
- Semantic tests: `test/Analysis/memory-tokens*.mlir`
- Physical tests: `test/Transforms/insert-sync*.mlir`
- Encoding tests: `test/Emit/sync.mlir`

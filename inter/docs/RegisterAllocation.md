# Register Allocation

## Role and phase boundary

Register allocation assigns physical ARF and GRF storage after machine
scheduling and before SWSB insertion. Relative storage constraints from tuples,
DPAS accumulator ties, fixed registers, and structured control flow are explicit
in XeMachine IR.

The pipeline performs:

1. alias preparation;
2. pre-RA machine scheduling;
3. ARF state construction and allocation;
4. iterative GRF state construction, linear scan, and relief;
5. physical synchronization.

## Storage alias model

`RegisterStorageAliasOpInterface` provides weighted dword-offset relations.
Implemented relations include:

- tuple joins and splits;
- destructive tuple updates;
- DPAS destination/accumulator aliasing;
- structured region operands, arguments, yields, backedges, and results;
- repeated references to fixed architectural GRFs.

Alias analysis builds connected components, assigns every member a normalized
dword offset, computes component extent, and derives a fixed origin when any
member is physical. Conflicting physical origins or inconsistent weighted cycles
are hard errors.

## Preparation

Preparation runs before scheduling and defensively before every GRF allocation
state rebuild. It is designed to be idempotent.

### Immediate legalization

An i64 immediate used by a non-move ALU operation is materialized in a virtual
GRF so later encoding does not face an illegal source form.

### Tuple repair

Preparation inserts marked moves when tuple elements are duplicated, used in
incompatible slots, imply inconsistent fixed origins, form misaligned shifted
views, or otherwise cannot share one legal storage component. Exact aligned
views remain zero-cost aliases.

### Destructive updates

An `update_tuple` base is copied when repeated execution or later liveness makes
in-place overwrite unsafe. Update values are placed directly only when their
storage, defining instruction, offset, and lifetime permit the destructive
write. Otherwise an `update-value` copy writes the required destination slot.

Sub-GRF placement is represented by instruction destination subregisters and
alias offsets, not by a sub-GRF allocation unit.

### Structured transfers

Preparation repairs branch joins, loop entries, and loop backedges. Crossing
parallel transfers are snapshotted before writes. Repetitive regions copy
live-through or duplicate initial state, and backedge swaps/overlaps are
materialized as safe parallel copies.

Region reachability and mutually exclusive alternatives come from
`XeMachineRegionFlow`.

## ARF allocation

ARFs are allocated before GRFs. Current virtual allocation supports only flag
storage:

- file `f`;
- width two dwords;
- physical choices f0 and f1.

Ranges use inclusive function-preorder positions and extend captures through
repetitive regions. Fixed ranges are processed before virtual ranges at equal
starts. Virtual flags use deterministic first-fit allocation.

Virtual address, accumulator, MME, and special-register files are rejected.

## GRF allocation state

Each attempt rebuilds positions, aliases, liveness, and components from current
IR. A component records:

- members and relative offsets;
- dword extent, rounded to allocation GRFs;
- inclusive start and end positions;
- optional fixed base;
- fixed-overlap permission;
- tentative or committed assignment.

Definition positions account for zero-byte forwarding. End positions include
direct uses, send payload lifetime through token completion, token forwarding,
region escape, and values captured by repetitive regions.

This is conservative linearized liveness, not path-sensitive CFG interference.

## Linear scan

Allocation uses deterministic first-fit whole-GRF linear scan:

1. Sort components by start, with fixed components first at equal starts.
2. Expire active components ending before the current start.
3. Place a fixed component only at its required base.
4. Scan virtual bases upward from `xemachine.reserved_grf_count`.
5. Require one contiguous interval large enough for the rounded component.
6. Reject overlap with active assignments and every temporally overlapping
   fixed component.
7. On success, rewrite every member's register type with its physical base.
8. On failure, record the failed component and position without partially
   committing assignments.

Explicit fixed storage may occupy the reserved prefix. A special fixed-overlap
marker permits overlap only for the marked fixed components.

## Transactional transform loop

GRF allocation state is serialized in the function attribute
`xemachine.regalloc_state`. One iteration runs:

1. `regalloc_build_state`;
2. `regalloc_linear_scan`;
3. `regalloc_remat_relief` after failure;
4. `regalloc_scratch_relief` if rematerialization did not apply.

A successful relief rewrite clears state and requests a complete rebuild.
Success removes temporary state and records iteration count. Failure that no
provider can relieve is a hard stall. The default maximum is 32 iterations.

The transaction protects tentative assignments; it is not a general IR rollback.
Relief rewrites and already completed ARF allocation remain in the IR.

## Rematerialization

Rematerialization considers one-result, non-fixed singleton components live
across the failure. The defining operation must be a supported cheap ALU
operation with no ARF operand. The candidate with the latest end is cloned at
eligible consumers after the failure and marked rematerialized.

There is no recursive cost model, component-size weighting, or live-range
splitting.

## Scratch spilling

Scratch is attempted only after rematerialization. It is disabled when ordinary
code defines `a0`, because scratch setup owns `a0.2`.

Eligible values are singleton, non-fixed, non-send results of exactly 16 or 32
dwords, live across the failure. The latest-ending candidate is selected.

Scratch slots are 64-byte aligned. Setup derives the scratch surface offset from
`r0`, installs it in `a0.2`, and is reused by later spills. A UGM scratch store
is inserted after the definition and drained with `sync allrd`; reloads are
inserted before consumers and reused within one owner operation.

Scratch code is not machine-scheduled after insertion.

## Validation

Before each GRF attempt, Inter validates:

- supported ALU execution sizes and element types;
- destination/source regions within declared storage;
- at most two GRFs per ordinary ALU operand;
- source rows that do not cross GRF boundaries;
- whole-GRF send payload/result widths and descriptor packet limits;
- named-message source/destination limits;
- tuple widths, alignment, offsets, and physical placement;
- DPAS A/B/C/D widths and destructive accumulator placement.

Allocation also rejects register-valued function signatures, remaining virtual
ARFs, invalid GRF budgets, and negative scratch sizes. After synchronization,
resource analysis rejects virtual GRFs and physical extents beyond the selected
GRF count.

There is no separate final pass that reconstructs and verifies all allocator
interference independently of allocation.

## Current limitations

- Allocation bases and component extents are whole-GRF units.
- Sub-GRF support is limited to prepared alias offsets and instruction fields.
- Liveness is conservative preorder-position based.
- Only f0/f1 virtual flags are allocated.
- No live-range splitting or SLM spill provider exists.
- Scratch supports singleton 16/32-dword values only.
- Send results, tuple bundles, DPAS chains, and arbitrary widths cannot spill.
- Relief selection uses latest end rather than a full cost model.
- Scratch setup reuse is function-wide and has limited dominance reasoning.
- No post-relief scheduler runs.

## Normative sources and tests

- Alias analysis: `include/inter/Dialect/XeMachine/IR/XeMachineAliasAnalysis.h`
  and `lib/inter/Dialect/XeMachine/IR/XeMachineAliasAnalysis.cpp`
- Preparation: `lib/inter/Dialect/XeMachine/IR/XeMachineRegAllocPreparation.cpp`
- Allocation: `lib/inter/Dialect/XeMachine/IR/XeMachineRegAlloc.cpp`
- Transform API: `include/inter/Dialect/XeMachine/IR/XeMachineTransformOps.td`
- Preparation tests: `test/Transforms/prepare-regalloc*.mlir`
- Allocation tests: `test/Transforms/regalloc*.mlir`
- Footprint tests: `test/Transforms/regalloc-invalid-footprint.mlir`
- Resource tests: `test/Transforms/resource-info*.mlir`

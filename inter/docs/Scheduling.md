# Machine Scheduling

## Role and phase boundary

Inter schedules XeMachine after instruction selection, tuple/payload
canonicalization, LICM, and register-allocation preparation. Scheduling runs
before ARF and GRF allocation. It fixes the order consumed by allocation and
physical synchronization.

There is one machine scheduler. Instructions inserted later by rematerialization
or scratch spilling are not rescheduled.

The implementation is split between:

- target-neutral graph construction and movement in `MachineScheduler.cpp`;
- BMG/Xe2 policy in `Xe2ScheduleModel.cpp`;
- timing data in `Xe2Timing.cpp`.

## Scheduled regions

The pass collects all regions before mutating IR. A scheduling segment is a
contiguous block range of schedulable operations with no nested regions.

Segments end at:

- terminators;
- full-scoreboard drains and EOT;
- unsupported or non-issue operations;
- every nested-region operation.

The scheduler recursively handles `exec_if`, `uniform_if`, `uniform_loop`, and
`payload_prologue`. Other nested-region operations fail. Each segment receives
fresh issue and pressure state; timing is not carried across boundaries.

The current pass accepts only functions with `xemachine.target = "bmg"`.

## Dependency graph

### SSA dependencies

Every operand whose definition is inside the segment creates an edge.

- Ordinary value dependencies are RAW.
- `!xemachine.mem.token` dependencies are ordering edges.

The scheduler does not query memory effects or alias analysis. Memory operations
may reorder unless semantic token inference and selection created an SSA edge.

### Physical storage hazards

Known physical GRFs and ARFs create RAW, WAR, and WAW edges. The model records
whole GRFs touched by a register extent and uses fixed origins from alias
components when available. Purely virtual components do not create physical
storage hazards at this pre-allocation stage.

Hazard tracking is GRF-granular, not byte/subregister-granular.

### Destructive and live-out storage

Definitions whose alias chain reaches a partial destructive update or escapes
the segment are pinned against earlier operations. Full-width, zero-offset
destructive continuations are followed so the chain endpoint, rather than every
ancestor, receives the conservative ordering.

### Loop-carried storage

For `uniform_loop`, users of carried inputs receive WAR edges to physical or
aliased backedge definitions. This prevents destructive backedge placement from
overwriting a value before its current-iteration reads complete.

If adding a target edge would oppose an existing path, the segment is left in
its original order. A cycle encountered by actual ready-set scheduling is a
hard error.

## Scheduling algorithm

The algorithm is a deterministic original-order gap filler, not a general list
scheduler.

For each segment:

1. Build the dependency graph and ready set.
2. Drain ready zero-byte operations to closure in original order.
3. Choose the first unscheduled original-order operation as the baseline.
4. If the baseline is ready and predicts no stall, keep it.
5. Otherwise scan ready candidates in original order.
6. Project the candidate plus newly ready zero-byte operations.
7. Reject a projection that violates pressure policy.
8. Accept the first candidate that issues before the stalled baseline without
   delaying the baseline.
9. Commit timing state and continue.
10. Apply the new order only if final pressure remains acceptable.

A full destructive continuation that already stalls may interleave with another
such continuation when it fills an earlier part of the baseline's stall and
does not delay the baseline.

Zero-byte tuple and token operations propagate readiness without consuming an
issue cycle.

## Xe2 timing model

Every machine instruction is classified by issue class and pipe. Current
classes cover moves/logic, arithmetic, accumulator arithmetic, ARF writes,
sends, sync, and systolic operations. Pipes are integer, floating, send,
systolic, or none.

The model keeps completion latency, pipe occupancy, and send-source read latency
separate. Representative current values are:

- move/logic: 10 cycles;
- arithmetic: 10 plus execution-width scale;
- accumulator arithmetic: 6 plus width scale;
- ARF write: 16;
- DPAS: 22-33 cycles based on repeat count;
- SLM load/store: 28 cycles at SIMD16, 45 at SIMD32;
- untyped L1/L3: 45/200;
- barrier: 30.

Send source-read latency is eight cycles plus payload GRF count. Exact constants
and classifications in `Xe2Timing.cpp` are normative.

Required gaps are:

- RAW: maximum of completion latency and occupancy;
- WAR/WAW: maximum of source-read latency, or two cycles for non-sends, and
  occupancy;
- token order: source-read latency when present, otherwise occupancy.

Issue time is the maximum of current cycle, predecessor readiness, and selected
pipe availability.

## Pressure policy

Pressure is computed over immutable register-alias components. Each component
contributes its dword extent rounded up to whole GRFs. The model records
in-segment definitions/uses, live-in/live-after state, external definitions,
zero-byte forwarding, and send payload lifetime through token completion.

The original-order peak is the baseline. Candidate movement is accepted only
when it does not exceed that peak.

This is aggregate whole-GRF pressure. It does not model fragmentation, exact
subregister liveness, fixed-register placement pressure, or the reserved-GRF
prefix separately.

## Invariants

- Memory ordering comes only from token SSA.
- Required storage and loop-carried hazards cannot be dropped.
- Target preview is side-effect free; commit mutates timing state.
- Candidate selection is deterministic by original operation index.
- Final operation order is established before SWSB insertion.
- Running the scheduler twice is expected to be stable.

## Current limitations

- BMG is the only timing target.
- There is no post-allocation or post-spill scheduling pass.
- Physical hazards and pressure are whole-GRF-granular.
- Timing state restarts at every segment boundary.
- The model does not account for SBID pressure, DPAS read suppression, cache
  behavior, or SLM banks.
- Liveness and pressure are conservative region-aware linear models, not a
  path-sensitive global scheduler.
- A rejected speculative edge can silently preserve original order rather than
  produce an optimization remark.

## Normative sources and tests

- API: `include/inter/Transforms/MachineScheduler.h`
- Generic algorithm: `lib/inter/Transforms/MachineScheduler.cpp`
- Xe2 policy: `lib/inter/Transforms/Xe2ScheduleModel.cpp`
- Timing: `lib/inter/Dialect/XeMachine/IR/Xe2Timing.cpp`
- Pass integration: `lib/inter/Transforms/InterMachineSchedule.cpp`
- Scheduler tests: `test/Transforms/machine-schedule*.mlir`
- Timing tests: `test/Analysis/xe2-timing*.mlir`

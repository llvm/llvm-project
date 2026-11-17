# ClangIR Points-to Analysis


## Introduction

This document describes the points-to analysis for CIR. For each pointer value
in a function, the analysis computes the set of objects the pointer may point
to. Transformations can use these sets to tell which objects a memory access
may touch, and which objects may be reached by code the analysis does not see.

The analysis is in an early stage. It tracks only a small set of operations,
and it is meant to grow in small steps. The last part of this document
describes what it handles today and what is expected to come next.

The analysis is implemented in
`clang/lib/CIR/Dialect/Analysis/CIRPointsToAnalysis.cpp`.

## Relation to Alias Analysis

CIR also has alias analysis, built on MLIR's `mlir::AliasAnalysis`. Its basic
implementation, `CIRBasicAliasAnalysis`, walks a pointer back through address
computations such as `cir.get_member` and `cir.ptr_stride` to the object it
comes from. It stops at loads, so two pointers loaded from memory may always
alias as far as it can tell. This is common in CIR, because Clang emits a
`cir.alloca` for every local variable, including local pointer variables, and
every use of such a variable loads the pointer from its slot.

The points-to analysis follows pointers through memory instead, which is the
part the basic alias analysis cannot see. The two are meant to complement each
other. A later change can register an alias analysis implementation based on
the points-to sets after the basic one, so that it is only asked about the
pairs the basic analysis cannot answer.

## Objects and Points-to Sets

An object is identified by the operation that allocates it. Currently only
`cir.alloca` objects are tracked, and an object is represented by the
`cir.alloca` result. A points-to set holds tracked objects and may also contain
"unknown", which stands for any object the analysis does not track: memory
reached through a function argument or a global, memory written by code the
analysis does not see, and tracked objects whose address escaped.

The analysis keeps two kinds of sets:

- for each value, the objects the value may point to,
- for each tracked object, the objects its memory may point to, that is, what
  a load from the object may return.

For example:

```c
int a;
int *p = &a;
int **pp = &p;
```

Each variable is a `cir.alloca` object: `a`, `p` and `pp`. The address of each
variable points to the variable itself, so the `cir.alloca` result for `p`
points to `{p}`. The memory sets are `p -> {a}` and `pp -> {p}`. Loading from
`pp` gives a value that points to `{p}`, and loading again through that value
gives a value that points to `{a}`.

## Computing the Sets

The analysis is inclusion based and flow insensitive. Every operation adds
members to some sets, and sets never lose members. The order of operations and
the control flow between them are not taken into account: a pointer stored
into an object anywhere in the function may be loaded back from it anywhere in
the function.

The rules are:

- A `cir.alloca` result points to the object it allocates.
- A null pointer constant points to no object.
- `cir.store %v, %p`: for every object `%p` points to, the memory of that object
  may point to everything `%v` points to.
- `%v = cir.load %p`: `%v` points to everything the memory of the objects `%p`
  points to may point to.
- Any other value, such as a function argument or the result of an operation
  the analysis does not model, points to unknown. So does a value defined
  outside the operation the analysis was computed for.
- If `%p` points to unknown, a load through `%p` returns unknown and a store
  through `%p` makes the stored objects escape.
- Any other use of a value makes the objects it points to escape.

When an object escapes, code the analysis does not see may access it, so its
memory is assumed to point to unknown in addition to what was stored in it, and
every object stored in it escapes as well.

The rules are applied to every operation, repeatedly, until no set changes.
Because sets only grow, this terminates.

## Limits

The analysis has limits on its size and on the number of times it visits
operations, so that large functions don't make compilation slow. The limits
are given by `PointsToLimits`:

- The number of operations. As soon as it finds more than `maxOperations`
  (5000) operations nested in the operation it is computed for, the analysis
  gives up.
- The work to compute the sets, counted as the number of times an operation is
  visited. Every operation is visited again each time a set changed during the
  previous visit of the function. The limit is `workPerOperation` (32) visits
  for each operation, and at least `minWork` (100000) visits in total. When it
  is reached, the analysis gives up.
- The size of a set. A set that grows past `maxSetSize` (100) objects is
  replaced by unknown. The analysis no longer knows which objects the set held,
  so they escape. Because a visit merges sets, this also bounds the work of a
  single visit.

When the analysis hits the operation or the work limit, it drops every set, and
every value points to unknown. `hitLimit()` tells whether that happened.

## Using the Results

`PointsToAnalysis` is built for an operation, usually a function, and computes
all the sets in its constructor. `getPointsTo(value)` returns the set of a
value. Values the analysis did not compute a set for point to unknown.

```c++
cir::PointsToAnalysis pointsTo(funcOp);
const cir::PointsToSet &set = pointsTo.getPointsTo(ptr);
if (!set.containsUnknown())
  for (mlir::Value object : set.getObjects())
    ...
```

Unknown may stand for a tracked object whose address escaped. The analysis
doesn't tell yet which objects escaped, so a set that contains unknown has to
be treated as possibly pointing to any object, including the tracked ones.

A set with no object and no unknown belongs to a null pointer, or to a value
loaded from memory that only null pointers, or nothing, were stored to.

A set can be printed with `<<`, which prints the names of its objects, sorted,
followed by unknown if the set contains it, for example `[a, b, unknown]`.

The sets describe the IR at the time the analysis is built and are not updated
afterwards. A pass that changes the IR in a way that affects how pointers flow
must build a new analysis before using it again.

For example:

```c
void f(void) {
  int a, b;
  int *p = &a, *q = &b;
  *p = 1;
  *q = 2;
}
```

```mlir
%1 = cir.alloca "a" align(4) : !cir.ptr<!s32i>
%2 = cir.alloca "b" align(4) : !cir.ptr<!s32i>
%3 = cir.alloca "p" align(8) init : !cir.ptr<!cir.ptr<!s32i>>
%4 = cir.alloca "q" align(8) init : !cir.ptr<!cir.ptr<!s32i>>
cir.store align(8) %1, %3 : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
cir.store align(8) %2, %4 : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
%6 = cir.load deref align(8) %3 : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
cir.store align(4) %5, %6 : !s32i, !cir.ptr<!s32i>
%8 = cir.load deref align(8) %4 : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
cir.store align(4) %7, %8 : !s32i, !cir.ptr<!s32i>
```

The two stores go through `%6` and `%8`, which are load results, so the basic
alias analysis cannot tell them apart. The memory of `p` only ever holds `a`
and the memory of `q` only ever holds `b`, so `%6` points to `{a}` and `%8`
points to `{b}`: the two stores write to different objects.

## Remarks

The analysis reports its results through MLIR's optimization remarks, in the
`points-to` category. They are only emitted when remarks are enabled for that
category, for example with `--remarks-filter=points-to` in `cir-opt`, and cost
nothing otherwise. There are two kinds:

- Analysis remarks give the set of every pointer loaded in the function, on the
  `cir.load` that loads it.
- Missed remarks tell where the analysis lost precision: each object whose
  address escapes, on the operation that first made it escape, each set that
  grew past the size limit and was replaced by unknown, and the operation or
  work limit, when one is hit.

For the example above, in a file `f.c`, `cir-opt --remarks-filter=points-to`
reports:

```
f.c:4:4: remark: [Analysis] PointsTo | Category:points-to | Function=f | Remark="points to [a]", RemarkId=1
f.c:5:4: remark: [Analysis] PointsTo | Category:points-to | Function=f | Remark="points to [b]", RemarkId=2
```

`--remarks-filter-analyse` and `--remarks-filter-missed` select only one of the
two kinds.

## Current Limitations and Next Steps

The analysis is deliberately small, and conservative wherever it does not model
something. The following are expected to be added, roughly in this order, each
as its own change:

- Following address computations (`cir.get_member`, `cir.ptr_stride`,
  `cir.get_element`, casts), so that taking the address of a member or an
  element does not make the whole object escape.
- Telling users of the analysis which objects escaped, so that an object whose
  address never escapes can be told apart from what unknown stands for. Today
  this is only reported through remarks.
- Calls, using the side effect attributes of `cir.call` and the copy and move
  semantics of special member functions.
- Values passed through region results and block arguments.
- Heap allocations and globals as tracked objects.
- An alias analysis implementation based on the points-to sets, registered
  after `CIRBasicAliasAnalysis`, as described above.
- A solver that only revisits the operations affected by a change, instead of
  every operation.

## Testing

The analysis is tested with the `test-cir-points-to` pass of `cir-opt`, which
computes the analysis for each function, and with the remarks it emits, checked
with `-verify-diagnostics`. CIR generated from C++ sources is tested the same
way, with the expected remarks written next to the source lines. The
`max-operations`, `min-work`, `work-per-operation` and `max-set-size` options
of the pass override the limits.

The tests are in `clang/test/CIR/Analysis/`, and the test pass is in
`clang/test/CIR/lib/Analysis/`.

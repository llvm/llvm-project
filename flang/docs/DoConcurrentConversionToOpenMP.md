<!--===- docs/DoConcurrentConversionToOpenMP.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

# `DO CONCURRENT` mapping to OpenMP

```{contents}
---
local:
---
```

This document describes the effort to parallelize `do concurrent` loops by
mapping them to OpenMP worksharing constructs. The goals of this document
are:
* Describing how to instruct `flang` to map `DO CONCURRENT` loops to OpenMP
  constructs.
* Tracking the current status and limitations of such mapping.
* Describing next steps.

## Usage

In order to enable `do concurrent` to OpenMP mapping, `flang` adds a new
compiler flag: `-fdo-concurrent-to-openmp`. This flag has 3 possible values:
1. `host`: this maps `do concurrent` loops to run in parallel on the host CPU.
   This maps such loops to the equivalent of `omp parallel do`.
2. `device`: this maps `do concurrent` loops to run in parallel on a target device.
   This maps such loops to the equivalent of
   `omp target teams distribute parallel do`.
3. `none`: this disables `do concurrent` mapping altogether. In that case, such
   loops are emitted as sequential loops.

The `-fdo-concurrent-to-openmp` compiler switch is currently available only when
OpenMP is also enabled. So you need to provide the following options to flang in
order to enable it:
```
flang ... -fopenmp -fdo-concurrent-to-openmp=[host|device|none] ...
```
For mapping to device, the target device architecture must be specified as well.
See `-fopenmp-targets` and `--offload-arch` for more info.

## Current status

Since LLVM 22, flang has more extensive support for parallelizing `do concurrent`
loops on the CPU and the GPU. In particular, the `local` specifier, partial
support for `reduce`, and automatic mapping of user-defined types are supported.
On the CPU, we validated the feature using [FIATS](https://github.com/BerkeleyLab/fiats)
inference and training codes where `do concurrent` and OpenMP had very similar
acceleration results (for more information, see: [1]).
On the GPU, we have basic support that is still in progress. We validated using
codes that do not make extensive use of user-defined types and allocatables.

[1] Automatically Parallelizing Batch Inference on Deep Neural Networks Using Fiats
and Fortran 2023 “Do Concurrent” (https://link.springer.com/chapter/10.1007/978-3-032-07612-0_11)

### Support summary

| Feature                                     | Host                | Device              |
|---------------------------------------------|---------------------|---------------------|
| Single-range loops                          | Supported           | Supported           |
| Multi-range loops (mapped as collapsed)     | Supported           | Supported           |
| `LOCAL` locality specifier                  | Supported           | Supported           |
| `LOCAL_INIT` locality specifier             | Not supported       | Not supported       |
| `REDUCE` locality specifier (Fortran 2023)  | Partial             | Partial             |
| Nested `do concurrent` constructs           | Outermost loop only | Outermost loop only |
| Derived types with allocatable components   | n/a                 | Supported           |
| Non-rectangular loop nests                  | Not supported       | Not supported       |

Notes on the above:
* `LOCAL_INIT` is rejected with a "not yet implemented" error when the pass
  encounters it.
* All standard reduction operations are mapped: `+`, `*`, `.and.`, `.or.`,
  `.eqv.`, `.neqv.`, `max`, `min`, `iand`, `ior`, and `ieor`. Support has so
  far been exercised mostly with scalar reduction variables, hence "partial".
* For nested `do concurrent` constructs, only the outermost construct is
  parallelized; inner constructs are emitted as sequential loops (see below).
* On the device, derived types with allocatable components are mapped through
  implicitly-generated `omp.declare_mapper` ops, including nested derived
  types. Derived types with pointer components are not handled yet.
* Implicit mapping detection (for mapping to the target device) is still quite
  limited and work to make it smarter is underway for both OpenMP in general
  and `do concurrent` mapping.

## Loop representation in FIR

`do concurrent` constructs are modeled by two dedicated FIR operations:
* `fir.do_concurrent`: a wrapper op whose region contains the allocations of
  the iteration variables followed by the loop op itself.
* `fir.do_concurrent.loop`: the loop op. It carries the bounds of **all**
  iteration ranges of the construct as well as the construct's locality
  specifiers (`local(...)` and `reduce(...)` operand groups).

For example, the following loop:
```fortran
  do concurrent(i=1:n, j=1:m)
    a(i,j) = i * j
  end do
```
is represented as:
```
fir.do_concurrent {
  %i = fir.alloca i32 {bindc_name = "i"}
  %i_decl:2 = hlfir.declare %i ...
  %j = fir.alloca i32 {bindc_name = "j"}
  %j_decl:2 = hlfir.declare %j ...

  fir.do_concurrent.loop (%i_iv, %j_iv) = (%i_lb, %j_lb) to (%i_ub, %j_ub) step (%i_st, %j_st) {
    ... loop body goes here ...
  }
}
```

Since a single `fir.do_concurrent.loop` op carries all the iteration ranges of
a multi-range construct, the conversion pass maps such loops directly to
"collapsed" OpenMP loops; there is no need to pattern-match nests of
single-range loop ops.

### Nested `do concurrent` constructs

Nested constructs are a different matter: given the following loop nest, even
though both loops are separate `do concurrent` constructs, only the outer loop
is parallelized:
```fortran
do concurrent(i=1:n)
  do concurrent(j=1:m)
    a(i,j) = i * j
  end do
end do
```

The pass skips any construct nested inside another one it converts. The inner
constructs are later lowered to sequential loops that run within each iteration
of the parallelized outer loop. Combining such nests into a single collapsed
OpenMP loop is possible future work (see the "Next steps" section).

See `flang/test/Transforms/DoConcurrent/not_perfectly_nested.f90` and
`flang/test/Transforms/DoConcurrent/skip_all_nested_loops.f90` for examples
of how nested constructs are handled.

## Mapping examples

### Single-range loops

Given the following loop:
```fortran
  do concurrent(i=1:n)
    a(i) = i * i
  end do
```

#### Mapping to `host`

Mapping this loop to the `host`, generates MLIR operations of the following
structure:

```
%4 = fir.address_of(@_QFEa) ...
%6:2 = hlfir.declare %4 ...

omp.parallel {
  // Allocate private copy for `i`.
  %19 = fir.alloca i32 {bindc_name = "i"}
  %20:2 = hlfir.declare %19 {uniq_name = "_QFEi"} ...

  omp.wsloop {
    omp.loop_nest (%arg0) : index = (%21) to (%22) inclusive step (%c1_2) {
      %23 = fir.convert %arg0 : (index) -> i32
      // Use the privatized version of `i`.
      fir.store %23 to %20#1 : !fir.ref<i32>
      ...

      // Use "shared" SSA value of `a`.
      %42 = hlfir.designate %6#0
      hlfir.assign %35 to %42
      ...
      omp.yield
    }
  }
  omp.terminator
} {omp.combined}
```

#### Mapping to `device`

Mapping the same loop to the `device` generates the equivalent of an
`omp target teams distribute parallel do` construct. Variables live into the
loop are mapped into the target region with implicit `omp.map.info` ops and the
loop bounds are evaluated on the host and passed to the region through the
`host_eval` interface:

```
%i_map = omp.map.info var_ptr(%i_decl#1 ...) map_clauses(implicit, ...) ...
%a_bounds = omp.map.bounds lower_bound(%c0) upper_bound(...) extent(...) ...
%a_map = omp.map.info var_ptr(%a_decl#1 ...) map_clauses(implicit, tofrom)
           capture(ByRef) bounds(%a_bounds) ...

omp.target kernel_type(spmd)
    host_eval(%lb_host -> %lb, %ub_host -> %ub, %step_host -> %step : ...)
    map_entries(%i_map -> %i_arg, %a_map -> %a_arg, ...) {
  %a_dev:2 = hlfir.declare %a_arg ...
  omp.teams {
    omp.parallel {
      // Allocate private copy for `i`.
      %25 = fir.alloca i32 {bindc_name = "i"}
      %26:2 = hlfir.declare %25 {uniq_name = "_QFEi"} ...

      omp.distribute {
        omp.wsloop {
          omp.loop_nest (%arg0) : index = (%lb) to (%ub) inclusive step (%step) {
            ... loop body using the privatized `i` and the mapped `a` ...
            omp.yield
          }
        }
      }
      omp.terminator
    }
    omp.terminator
  }
  omp.terminator
}
```

### Multi-range loops

The pass supports multi-range loops as well. Given the following
example:

```fortran
   do concurrent(i=1:n, j=1:m)
       a(i,j) = i * j
   end do
```

The generated `omp.loop_nest` operation collapses both ranges into a single
worksharing loop:

```
omp.loop_nest (%arg0, %arg1)
    : index = (%17, %19) to (%18, %20)
    inclusive step (%c1_2, %c1_4) collapse(2) {
  fir.store %arg0 to %private_i#1 : !fir.ref<i32>
  fir.store %arg1 to %private_j#1 : !fir.ref<i32>
  ...
  omp.yield
}
```

It is worth noting that we have privatized versions for both iteration
variables: `i` and `j`. These are locally allocated inside the parallel/target
OpenMP region similar to what the single-range example in previous section
shows.

## Data environment

By default, variables that are used inside a `do concurrent` loop nest are
either treated as `shared` in case of mapping to `host`, or mapped into the
`target` region using a `map` clause in case of mapping to `device`. The only
exceptions to this are:
  1. the loop's iteration variable(s) (IV). For each IV, we allocate a local
     copy as shown by the mapping examples above.
  1. any values that are from allocations outside the loop nest and used
     exclusively inside of it. In such cases, a local privatized
     copy is created in the OpenMP region to prevent multiple teams of threads
     from accessing and destroying the same memory block, which causes runtime
     issues. For an example of such cases, see
     `flang/test/Transforms/DoConcurrent/locally_destroyed_temp.f90`.
  1. variables with locality specifiers, described below.

### Locality specifiers

`LOCAL` specifiers are supported for both host and device mapping. On the FIR
level, a `LOCAL` variable is modeled by a `fir.local` "localizer" op (which,
similar to OpenMP's `omp.private` op, can carry `init` and `dealloc` regions)
referenced from the `local(...)` operand group of the `fir.do_concurrent.loop`
op. The conversion pass translates each localizer into an equivalent
`omp.private` op and attaches the corresponding `private(...)` clause to the
generated worksharing loop. For examples, see
`flang/test/Transforms/DoConcurrent/locality_specifiers_simple.mlir` and
`flang/test/Transforms/DoConcurrent/local_device.mlir`.

`LOCAL_INIT` specifiers are not supported yet; the pass emits a
"not yet implemented" error when it encounters one.

### Reductions

Fortran 2023 `REDUCE` specifiers are mapped to OpenMP reductions for both host
and device. On the FIR level, a reduction is modeled by a
`fir.declare_reduction` op referenced from the `reduce(...)` operand group of
the `fir.do_concurrent.loop` op. The pass translates it into an equivalent
`omp.declare_reduction` op and attaches `reduction(...)` clauses to the
generated OpenMP ops: on the host, to the `omp.wsloop` op; on the device, to
both the `omp.teams` and the `omp.wsloop` ops.

When mapping to the device, reduction variables are mapped `tofrom` and
captured by reference so that the reduced values are copied back to the host
after the target region finishes.

For examples, see the `reduce_*` tests in
`flang/test/Transforms/DoConcurrent/`.

## Next steps

This section describes some of the open questions/issues that are not tackled
yet.

### `LOCAL_INIT` locality specifiers

The `fir.local` op models `LOCAL_INIT` localizers with a `copy` region, but
the conversion pass does not translate them to OpenMP firstprivate-like
privatizers yet.

### Combining nested `do concurrent` constructs

As pointed out earlier, a `do concurrent` construct nested inside another one
is not combined with its parent into a single collapsed OpenMP loop; only the
outermost construct is parallelized. Detecting when such combining is legal
(e.g. when there is no intervening code with side effects between the two
constructs) and profitable still needs to be implemented.

### Data-dependence analysis

Right now, we map loop nests without analysing whether such mapping is safe to
do or not. We probably need to at least warn the user of unsafe loop nests due
to loop-carried dependencies.

### Non-rectangular loop nests

So far, we did not need to use the pass for non-rectangular loop nests. For
example:
```fortran
do concurrent(i=1:n)
  do concurrent(j=i:n)
    ...
  end do
end do
```
We defer this to the (hopefully) near future when we get the conversion in a
good shape for the samples/projects at hand.

### Smarter implicit mapping to the device

Implicit mapping of live-in values into the target region covers the common
cases (scalars, arrays with compile-time or runtime shapes, allocatables, and
derived types with allocatable components) but still has gaps; for example,
derived types with pointer components are not handled yet. There is also no
way for the user to control the mapping behavior of individual variables.

### Generalizing the pass to other parallel programming models

Once we have a stable and capable `do concurrent` to OpenMP mapping, we can take
this in a more generalized direction and allow the pass to target other models;
e.g. OpenACC. This goal should be kept in mind from the get-go even while only
targeting OpenMP.

## Tests

The FIR-to-OpenMP conversion is tested in
`flang/test/Transforms/DoConcurrent/`, which contains both host and device
mapping tests.

In addition, the following end-to-end offloading tests execute
device-mapped `do concurrent` loops on an actual target device:
* `offload/test/offloading/fortran/do-concurrent-to-omp-saxpy.f90`
* `offload/test/offloading/fortran/do-concurrent-to-omp-saxpy-2d.f90`
* `offload/test/offloading/fortran/do-concurrent-to-omp-min-reduce.f90`
* `offload/test/offloading/fortran/do-concurrent-to-omp-nested-derived-type.f90`

The compiler flag itself is tested in
`flang/test/Driver/do_concurrent_to_omp_cli.f90`.

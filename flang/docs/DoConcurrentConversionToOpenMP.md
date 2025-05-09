<!--===- docs/DoConcurrentMappingToOpenMP.md

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

This document seeks to describe the effort to parallelize `do concurrent` loops
by mapping them to OpenMP worksharing constructs. The goals of this document
are:
* Describing how to instruct `flang` to map `DO CONCURRENT` loops to OpenMP
  constructs.
* Tracking the current status of such mapping.
* Describing the limitations of the current implementation.
* Describing next steps.
* Tracking the current upstreaming status (from the AMD ROCm fork).

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

Under the hood, `do concurrent` mapping is implemented in the
`DoConcurrentConversionPass`. This is still an experimental pass which means
that:
* It has been tested in a very limited way so far.
* It has been tested mostly on simple synthetic inputs.

### Loop nest detection

On the `FIR` dialect level, the following loop:
```fortran
  do concurrent(i=1:n, j=1:m, k=1:o)
    a(i,j,k) = i + j + k
  end do
```
is modelled as a nest of `fir.do_loop` ops such that an outer loop's region
contains **only** the following:
  1. The operations needed to assign/update the outer loop's induction variable.
  1. The inner loop itself.

So the MLIR structure for the above example looks similar to the following:
```
  fir.do_loop %i_idx = %34 to %36 step %c1 unordered {
    %i_idx_2 = fir.convert %i_idx : (index) -> i32
    fir.store %i_idx_2 to %i_iv#1 : !fir.ref<i32>

    fir.do_loop %j_idx = %37 to %39 step %c1_3 unordered {
      %j_idx_2 = fir.convert %j_idx : (index) -> i32
      fir.store %j_idx_2 to %j_iv#1 : !fir.ref<i32>

      fir.do_loop %k_idx = %40 to %42 step %c1_5 unordered {
        %k_idx_2 = fir.convert %k_idx : (index) -> i32
        fir.store %k_idx_2 to %k_iv#1 : !fir.ref<i32>

        ... loop nest body goes here ...
      }
    }
  }
```
This applies to multi-range loops in general; they are represented in the IR as
a nest of `fir.do_loop` ops with the above nesting structure.

Therefore, the pass detects such "perfectly" nested loop ops to identify multi-range
loops and map them as "collapsed" loops in OpenMP.

#### Further info regarding loop nest detection

Loop nest detection is currently limited to the scenario described in the previous
section. However, this is quite limited and can be extended in the future to cover
more cases. At the moment, for the following loop nest, even though both loops are
perfectly nested, only the outer loop is parallelized:
```fortran
do concurrent(i=1:n)
  do concurrent(j=1:m)
    a(i,j) = i * j
  end do
end do
```

Similarly, for the following loop nest, even though the intervening statement `x = 41`
does not have any memory effects that would affect parallelization, this nest is
not parallelized either (only the outer loop is).

```fortran
do concurrent(i=1:n)
  x = 41
  do concurrent(j=1:m)
    a(i,j) = i * j
  end do
end do
```

The above also has the consequence that the `j` variable will **not** be
privatized in the OpenMP parallel/target region. In other words, it will be
treated as if it was a `shared` variable. For more details about privatization,
see the "Data environment" section below.

See `flang/test/Transforms/DoConcurrent/loop_nest_test.f90` for more examples
of what is and is not detected as a perfect loop nest.

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
  // TODO Use delayed privatization.
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
    omp.terminator
  }
  omp.terminator
}
```

#### Mapping to `device`

<!-- TODO -->

### Multi-range loops

The pass currently supports multi-range loops as well. Given the following
example:

```fortran
   do concurrent(i=1:n, j=1:m)
       a(i,j) = i * j
   end do
```

The generated `omp.loop_nest` operation look like:

```
omp.loop_nest (%arg0, %arg1)
    : index = (%17, %19) to (%18, %20)
    inclusive step (%c1_2, %c1_4) {
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

### Data environment

By default, variables that are used inside a `do concurrent` loop nest are
either treated as `shared` in case of mapping to `host`, or mapped into the
`target` region using a `map` clause in case of mapping to `device`. The only
exceptions to this are:
  1. the loop's iteration variable(s) (IV) of **perfect** loop nests. In that
     case, for each IV, we allocate a local copy as shown by the mapping
     examples above.
  1. any values that are from allocations outside the loop nest and used
     exclusively inside of it. In such cases, a local privatized
     copy is created in the OpenMP region to prevent multiple teams of threads
     from accessing and destroying the same memory block, which causes runtime
     issues. For an example of such cases, see
     `flang/test/Transforms/DoConcurrent/locally_destroyed_temp.f90`.

Implicit mapping detection (for mapping to the target device) is still quite
limited and work to make it smarter is underway for both OpenMP in general 
and `do concurrent` mapping.

#### Non-perfectly-nested loops' IVs

For non-perfectly-nested loops, the IVs are still treated as `shared` or
`map` entries as pointed out above. This **might not** be consistent with what
the Fortran specification tells us. In particular, taking the following
snippets from the spec (version 2023) into account:

> § 3.35
> ------
> construct entity
> entity whose identifier has the scope of a construct

> § 19.4
> ------
>  A variable that appears as an index-name in a FORALL or DO CONCURRENT
>  construct [...] is a construct entity. A variable that has LOCAL or
>  LOCAL_INIT locality in a DO CONCURRENT construct is a construct entity.
> [...]
> The name of a variable that appears as an index-name in a DO CONCURRENT
> construct, FORALL statement, or FORALL construct has a scope of the statement
> or construct. A variable that has LOCAL or LOCAL_INIT locality in a DO
> CONCURRENT construct has the scope of that construct.

From the above quotes, it seems there is an equivalence between the IV of a `do
concurrent` loop and a variable with a `LOCAL` locality specifier (equivalent
to OpenMP's `private` clause). Which means that we should probably
localize/privatize a `do concurrent` loop's IV even if it is not perfectly
nested in the nest we are parallelizing. For now, however, we **do not** do
that as pointed out previously. In the near future, we propose a middle-ground
solution (see the Next steps section for more details).

<!--
More details about current status will be added along with relevant parts of the
implementation in later upstreaming patches.
-->

## Next steps

This section describes some of the open questions/issues that are not tackled yet
even in the downstream implementation.

### Separate MLIR op for `do concurrent`

At the moment, both increment and concurrent loops are represented by one MLIR
op: `fir.do_loop`; where we differentiate concurrent loops with the `unordered`
attribute. This is not ideal since the `fir.do_loop` op support only single
iteration ranges. Consequently, to model multi-range `do concurrent` loops, flang
emits a nest of `fir.do_loop` ops which we have to detect in the OpenMP conversion
pass to handle multi-range loops. Instead, it would better to model multi-range
concurrent loops using a separate op which the IR more representative of the input
Fortran code and also easier to detect and transform.

### Delayed privatization

So far, we emit the privatization logic for IVs inline in the parallel/target
region. This is enough for our purposes right now since we don't
localize/privatize any sophisticated types of variables yet. Once we have need
for more advanced localization through `do concurrent`'s locality specifiers
(see below), delayed privatization will enable us to have a much cleaner IR.
Once delayed privatization's implementation upstream is supported for the
required constructs by the pass, we will move to it rather than inlined/early
privatization.

### Locality specifiers for `do concurrent`

Locality specifiers will enable the user to control the data environment of the
loop nest in a more fine-grained way. Implementing these specifiers on the
`FIR` dialect level is needed in order to support this in the
`DoConcurrentConversionPass`.

Such specifiers will also unlock a potential solution to the
non-perfectly-nested loops' IVs issue described above. In particular, for a
non-perfectly nested loop, one middle-ground proposal/solution would be to:
* Emit the loop's IV as shared/mapped just like we do currently.
* Emit a warning that the IV of the loop is emitted as shared/mapped.
* Given support for `LOCAL`, we can recommend the user to explicitly
  localize/privatize the loop's IV if they choose to.

#### Sharing TableGen clause records from the OpenMP dialect

At the moment, the FIR dialect does not have a way to model locality specifiers
on the IR level. Instead, something similar to early/eager privatization in OpenMP
is done for the locality specifiers in `fir.do_loop` ops. Having locality specifier
modelled in a way similar to delayed privatization (i.e. the `omp.private` op) and
reductions (i.e. the `omp.declare_reduction` op) can make mapping `do concurrent`
to OpenMP (and other parallel programming models) much easier.

Therefore, one way to approach this problem is to extract the TableGen records
for relevant OpenMP clauses in a shared dialect for "data environment management"
and use these shared records for OpenMP, `do concurrent`, and possibly OpenACC
as well.

#### Supporting reductions

Similar to locality specifiers, mapping reductions from `do concurrent` to OpenMP
is also still an open TODO. We can potentially extend the MLIR infrastructure
proposed in the previous section to share reduction records among the different 
relevant dialects as well.

### More advanced detection of loop nests

As pointed out earlier, any intervening code between the headers of 2 nested
`do concurrent` loops prevents us from detecting this as a loop nest. In some
cases this is overly conservative. Therefore, a more flexible detection logic
of loop nests needs to be implemented.

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
good share for the samples/projects at hand.

### Generalizing the pass to other parallel programming models

Once we have a stable and capable `do concurrent` to OpenMP mapping, we can take
this in a more generalized direction and allow the pass to target other models;
e.g. OpenACC. This goal should be kept in mind from the get-go even while only
targeting OpenMP.


## Upstreaming status

- [x] Command line options for `flang` and `bbc`.
- [x] Conversion pass skeleton (no transormations happen yet).
- [x] Status description and tracking document (this document).
- [x] Loop nest detection to identify multi-range loops.
- [ ] Basic host/CPU mapping support.
- [ ] Basic device/GPU mapping support.
- [ ] More advanced host and device support (expaned to multiple items as needed).

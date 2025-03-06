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

<!--
More details about current status will be added along with relevant parts of the
implementation in later upstreaming patches.
-->

## Next steps

This section describes some of the open questions/issues that are not tackled yet
even in the downstream implementation.

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
- [ ] Basic host/CPU mapping support.
- [ ] Basic device/GPU mapping support.
- [ ] More advanced host and device support (expaned to multiple items as needed).

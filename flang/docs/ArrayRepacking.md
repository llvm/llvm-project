<!--===- docs/ArrayRepacking.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

# Assumed-shape arrays repacking
Fortran 90 introduced dummy arguments to be declared as assumed-shape arrays, which allowed to pass non-contiguous arrays to subprograms. In some cases, accessing non-contiguous arrays may result in poor program performance, and paying an overhead of copying a non-contiguous array into a contiguous memory (packing) before processing it may result in better performance. This document describes Flang compiler and runtime support for packing/unpacking of non-contiguous arrays.

## A problem case

[Example #1](#example-1) provides a way to compare performance of a repetitive access of a large array when the array is contiguous and non-contiguous. The `test` function remains the same in both cases to make sure that any difference in the code generation does not affect performance, and only the array layout in memory matters.

The example might be compiled using any Fortran 90 compiler, e.g. `gfortran -cpp example1.f90 -O2 <additional-options>`. The table below contains performance information for different compilations and targets:

| additional-options | AMD EPYC 9684X, GNU Fortran 13.2.0                           | Arm Neoverse V2, GNU Fortran 11.4.0                          |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `none`             | 20495466758      L1-dcache-loads<br/>            31403868      L1-dcache-prefetches<br/>     10116173649      L1-dcache-load-misses<br/>119,167,236,596      cycles | 20030549910      L1-dcache-loads<br/>   10233598442      L1-dcache-load-misses<br/>     1098681496      LLC-load-misses<br/>43,426,056,799      cycles |
| `-DREPACKING`      | 20245847735      L1-dcache-loads<br/>         583614040      L1-dcache-prefetches<br/>         644552282      L1-dcache-load-misses<br/>  10,837,843,298      cycles | 20023110457      L1-dcache-loads<br/>              294393      L1-dcache-load-misses<br/>              321878      LLC-load-misses<br/>10,065,421,618      cycles |
| `-frepack-arrays`  | 20248624699      L1-dcache-loads<br/>         584325700      L1-dcache-prefetches<br/>         644353154      L1-dcache-load-misses<br/>  10,850,830,504      cycles | 20023117997      L1-dcache-loads<br/>              275169      L1-dcache-load-misses<br/>              323902      LLC-load-misses<br/>10,066,689,166      cycles |

The default version is much slower than the version with manual array repacking and [-frepack-arrays](https://gcc.gnu.org/onlinedocs/gfortran/Code-Gen-Options.html#index-frepack-arrays) due to the L1 data cache misses even considering the extra overhead required to pack/unpack the non-contiguous array.

This artificial example was inspired by the innermost hot loop from `fourir` subroutine of [Polyhedron/capacita](https://fortran.uk/fortran-compiler-comparisons/the-polyhedron-solutions-benchmark-suite/) benchmark, which speeds up about 1.8x with GNU Fortran compiler's `-frepack-arrays` option on AMD EPYC 9684X and 1.3x - on Arm Neoverse V2.

Having these results it seems reasonable to provide support for arrays repacking in Flang compiler, which may reduce the amount of effort to rewrite existing Fortran programs for better data cache utilization.

## Implementations in other compilers

### GNU Fortran compiler

`-frepack-arrays` option of GNU Fortran compiler let's the compiler generate special subprogram prologue/epilogue code that performs automatic packing/unpacking of the assumed-shape dummy arrays. With some implementation limitations, the following happens for any such dummy array:

**In subprogram prologue:** iff the array is not contiguous in any dimension, it is copied into a newly allocated contiguous chunk of memory, and the following subprogram code operates on the temporary. This is a `pack` action consisting of the allocation and copy-in.

**In subprogram epilogue:** iff the array is not contiguous in any dimension, values from the temporary array are copied over to the original array and the temporary array is deallocated. This is `unpack` action consisting of the copy-out and deallocation. It makes sure any updates of the array done by the subprogram are propagated to the caller side.

#### Facts and guesses about the implementation

The dynamic checks for continuity and the array copy code is located completely in the [runtime](https://github.com/gcc-mirror/gcc/blob/3e08a4ecea27c54fda90e8f58641b1986ad957e1/libgfortran/generated/in_pack_r8.c#L35), so the compiler inserts unconditional calls in the subprogram prologue/epilogue.

It looks like `gfortran` ignores `intent(out)/intent(in)` which could have helped to avoid some of the `pack/unpack` overhead.

It looks like the `pack`/`unpack` actions are inserted early in the compilation pipeline, and these extra calls affect behavior of the later optimization passes. For example, `Polyhedron/fatigue2` slows down by about 2x with `-frepack-arrays`: this slowdown is not caused by the `pack`/`unpack` overhead, but is a consequence of worse function inlining decisions made after the calls insertion. The benchmarks becomes even faster than the original version with `-frepack-arrays` and proper `-finline-limit=` settings, but it does not look like the benchmark contains code that would benefit from the array repacking.

It does not look like `gfortran` is able to eliminate the `pack`/`unpack` code after the function inlining, if the actual argument is statically known to be contiguous. So the overhead from the dynamic continuity checks is inevitable when `-frepack-arrays` is specified.

It does not look like `gfortran` tries to optimize the insertion of `pack`/`unpack` code. For example, if a dummy array is only used under a condition within the subprogram, the repacking code might be inserted under the same condition to minimize the overhead on the unconditional path through the subprogram.

### NVIDIA HPC Fortran compiler

`nvfortran` compiler performs array repacking by default, and has few option to control this behavior (only [-M[no]target_temps](https://docs.nvidia.com/hpc-sdk/compilers/hpc-compilers-ref-guide/index.html#command-line-options-reference)). The compiler inserts `pack`/`unpack` code around the calls of subprograms that have assumed-shape dummy array arguments (a procedure having an assumed-shape dummy argument must have an explicit interface due to F2018 15.4.2.2, 1, (3), (b)):

**Before the call:** iff the array is not contiguous in the innermost dimension, it is copied into a newly allocated contiguous chunk of memory, and the temporary array is passed to the callee.

**After the call:** iff the array is not contiguous in the innermost dimension, values from the temporary array are copied to the original array and the temporary array is deallocated.

#### Facts and guesses about the implementation

The `pack` code is only generated if the actual argument may be non-contiguous in the innermost dimension, as determined statically, i.e. the compiler does not generate any dynamic continuity checks. For example:

```Fortran
interface
  subroutine test1(x)
    real :: x(:)
  end subroutine test1
  subroutine test2(x)
    real :: x(:,:)
  end subroutine test2
end interface
real :: x(m1,m2), y(1,m2)
call test1(x(1,:)) ! case 1
call test1(y(1,:)) ! case 2
call test2(x(1:n,:)) ! case 3
call test2(x(1:1,:)) ! case 4
```

In case 1, the `pack`/`unpack` code is generated without dynamically checking if `m1 == 1` (in which case the actual argument is actually contiguous).

In case 2, the `pack`/`unpack` code is also generated, which is a room for improvement.

In case 3 and 4, the `pack`/`unpack` code is not generated, because the actual argument is contiguous in the innermost dimension. There seems to be room for improvement in case 4, where it might be beneficial to repack the array in case `m1` is big enough to prevent the data cache utilization (depending on the actual processing of the array in `test2`, of course).

`nvfortran` does optimize out the `unpack` copy-out code in case the dummy argument is declared `intent(in)`, but it does not optimize the `pack` copy-in in case it is declared `intent(out)`.

It looks like `nvfortran` is not able to optimize the `pack`/`unpack` code after the function inlining (`-Minline=reshape`), even if the inline code makes it obvious that only a single element of the array is being accessed and there is no reason to copy-in/out the whole array.

`nvfortran`'s implementation guarantees that an assumed-shape dummy array is contiguous in the innermost dimension, so when such a dummy is passed to a callee as an actual argument associated with the callee's assumed-shape dummy array, there is no need to `pack`/`unpack` it again around the callee's call site.

## Known limitations of the array repacking

`gfortran` documentation, expectedly, warns that the array repacking `can introduce significant overhead to the function call, especially when the passed data is noncontiguous`. A compiler has to try to minimize the overhead of the copy-in/out actions whenever possible, but it may not be always possible to guess correctly when the repacking is profitable. So the `gfortran`'s approach of giving the users control over the repacking seems reasonable. A compiler may decide to enable array repacking by default or under some optimization levels, but the correctness issues described below has to be taken into account as well as performance and usability (i.e. the need to specify a compiler option to enable/disable array repacking).

**Difference between performance of nvfortran and gfortran **

Array repacking creates a complete copy of an array section and let's the program code work on the temporary copy, then reflecting the updates back through another copy. If the original program intends to let different threads to work on different parts of the same array section, then the copy-in/out actions introduce a data race that has not existed in the original program. [Example #2](#example-2) produces inconsistent results when being compiled with either `nvfortran -mp` or `gfortran -fopenmp -frepack-arrays` and run with multiple threads. Note that the `repacking` subroutine and its call site might be written such that they are located in separate modules that do not have to be compiled with `-mp/-fopenmp`, so a compiler has no clue whether array repacking is safe. Even if explicitly instructed via `-frepack-arrays`, the compiler cannot avoid false-positive warnings about unsafety of array repacking, because it cannot know whether a function might be called in a multithreaded context (e.g. when `-mp/-fopenmp` is not specified).

The array copies may also become a problem for OpenACC/OpenMP target data environment management. For example:

```Fortran
subroutine test(x)
  real :: x(:)
  !$acc serial present(x)
  ...
  !$acc end serial
end subroutine test
subroutine caller(n)
  integer :: n
  real :: x(n,n)
  !$acc enter data create(x)
  call test(x(1,:))
end subroutine caller
```

The whole array `x` is expected to be present in the device data environment after the `enter data` construct, but the actual array being "seen" at the `serial` construct is a temporary copy of the array section, which has no corresponding memory on the device.

A compiler could generate code that dynamically detect both of these situations, i.e. whether the point of repacking is happening in a multithreaded context or whether the array to be repacked has associated bookkeeping in the device data environment, and do not create copies. Such checks would introduce dependencies on the parallelization/offload runtime libraries, which are not linked unless compiler is instructed to do so via `-acc/-fopenacc/-mp/-fopenmp/etc.`

So it does not seem practical/reasonable to enable the array repacking by default in a compiler that must produce correct code for all standard conformant programs. It is still beneficial to let users request array repacking, given that its behavior is properly documented and all the warning signs are in place.

## Flang feature requirements

### Correctness

1. Support repacking of assumed-shape array dummy arguments or actual array arguments associated with such dummy arguments of any data types.
2. When array repacking is enabled, Flang should strive to provide correct program behavior when OpenACC/OpenMP features are explicitly enabled during the compilation.
   * It is unclear if the compiler/runtime can always prevent the array repacking to produce incorrect code. Thus, the implementation should do its best to prevent incorrect behavior or diagnose the incorrect behavior in runtime as soon as possible.
   * To stress again, such implementation can only be enabled when users enable OpenACC/OpenMP explicitly during the compilation, so that the compiler can rely on the appropriate OpenACC/OpenMP runtime libraries to be linked into the resulting module.

### Performance

1. Minimize the overhead of array repacking, e.g. avoid copy-in/out whenever possible, execute copy-in/out only on the execution paths where the array is accessed.
2. Provide different modes of repacking depending on the "continuity" meaning, i.e. one - array is contiguous in the innermost dimension, two - array is contiguous in all dimensions.
3. Avoid generating repacking code, when the "continuity" can be statically proven (including after optimization passes like constant propagation, function inlining, etc.).
4. Use a set of heuristics to avoid generating repacking code based on the array usage pattern, e.g. if an array is proven not to be used in an array expression or a loop, etc.
5. Use a set of heuristics to avoid repacking actions dynamically, e.g. based on the array size, element size, byte stride(s) of the [innermost] dimension(s), etc.
6. Minimize the impact of the IR changes, introduced by repacking, on the later optimization passes.

### Usability

1. Provide command line options to enable/disable array repacking, e.g. `-f[no-]repack-arrays` for `gfortran` cli compatibility.
2. Provide command line options to instruct the compiler which performance heuristics to use with the default picked based on benchmarking.
3. Provide consistent behavior of the temporary arrays with relation to `-fstack-arrays` (that forces all temporary arrays to be allocated on the stack).
4. Produce correct debug information to substitute the original array with the copy array when accessing values in the debugger.
5. Document potential correctness issues that array repacking may cause in multithreaded/offload execution.

## Proposed design

### Overview

Controlled by cli options, Lowering will generate a `fir.pack_array` operation in a subprogram's prologue for each assumed-shape dummy array argument (including `OPTIONAL`). For each `fir.pack_array` it will also generate `fir.unpack_array` in the subprogram's epilogue. These new operations will represent the complete effects of `pack`/`unpack` actions, such as temp-allocation/copy-in/copy-out/temp-deallocation. While it is possible to represent the needed actions using existing FIR/HLFIR operations, it is worth keeping them more specific and compact for easier manipulation in the passes related to optimizing the `pack`/`unpack` actions.

The new operations will hold all the information that customizes further handling of the `pack`/`unpack` actions, such as:

* Optional array of attributes supporting an interface to generate a predicate that says if the repacking is safe in the current context.
* The continuity mode: `innermost` vs `whole`.
* Attributes selecting the heuristics (both compiler and runtime ones) that may be applied to avoid `pack`/`unpack` actions.
* Other attributes, like `stack` vs `heap` to manage the temporary allocation according to `-fstack-arrays`, etc.

Lowering will not try to optimize the insertion of new operations, except for obvious cases like `CONTIGUOUS` dummy arguments or arrays of elements bigger than the element size threshold. Further optimization passes will be responsible for optimizing the operations away or moving them around to satisfy the performance requirements.

The following FIR passes should be implemented:

* Deletion of `fir.pack_array`/`fir.unpack_array` that are statically proven to take a contiguous input array.
* Deletion/merging of cascaded `fir.pack_array` operations.
* Deletion of the new operations that are statically proven not to meet the array usage patterns that are considered to benefit from the array repacking.
* Deletion of the new operations that are statically proven not to meet the dynamic conditions for repacking (such as the array size).
* Repositioning of `fir.pack_array`/`fir.unpack_array` to execution paths where the array is actually accessed.
* A pass converting the operations to the existing FIR operations and/or Fortran runtime calls.

### New operations to represent pack/unpack

#### fir.pack_array operation

The operation has the following syntax:

```
%new_var = fir.pack_array %var
    [stack ]
    [innermost ]
    [no_copy ]
    [heuristics([none|loop-only]) ]
    [constraints([max-size = <int>, ][max-element-size = <int>, ]
    			 [min-stride = <int>]) ]
    [typeparams %p1, ... ]
    [<[acc.temp_copy_is_safe][omp.temp_copy_is_safe]>]
    : !fir.box/class<!fir.array<...>>
```

The operation creates a new `!fir.box/class<!fir.array<>>` value to represent either the original `%var` or a newly allocated temporary array, maybe identical to `%var` by value.

Arguments:

* `stack` - indicates if `-fstack-arrays` is in effect for compiling this function.
* `innermost` - tells that the repacking has to be done iff the array is not contiguous in the innermost dimension. This also describes what type of continuity can be expected from `%new_var`, i.e. `innermost` means that the resulting array is definitely contiguous in the innermost dimension, but may be non-contiguous in other dimensions (unless additional analysis proves otherwise). For 1-D arrays, `innermost` attribute is not valid.
* `no_copy` - indicates that, in case a temporary array is created, `%var` to `%new_var` copy is not required (`intent(out)` dummy argument case).
* `heuristics`
  * `loop-only` - `fir.pack_array` can be optimized away, if the array is not used in a loop.
  * `none` - `fir.pack_array` cannot be optimized based on the array usage pattern.
* `constraints`
  * `max-size` - constant integer attribute specifying the maximum byte size of an array that is eligible for repacking.
  * `max-element-size` - constant integer attribute specifying the maximum byte element-size of an array that is eligible for repacking.
  * `min-stride` - constant integer attribute specifying the minimum byte stride of the innermost dimension of an array that is eligible for repacking.
* `typeparams` - type parameters of the element type.
* `*.temp_copy_is_safe`: a list of attributes implementing `TempCopyIsSafe` attribute interface for generating a boolean value indicating whether using a temporary copy instead of the original array is safe in the current context.

Memory effects are conservative, assuming that an allocation and copy may happen:

* `MemAlloc` effect on either `AutomaticAllocationScopeResource` or `DefaultResource` depending on `stack` attribute.
  * The memory allocation effect is especially important for operations with `stack` attribute so that `fir.pack_array` operations are not reordered inconsistently with their corresponding `fir.unpack_array` operations. This may cause issues with later lowering of this operations into `stacksave/stackrestore` pairs.
* If there is no `no_copy`:
  * `MemRead` effect on unknown value to indicate potential read from the original array. This effect prevents hoisting a `fir.pack_array` above any write to the original array.
    * [TBD] we can relax that by having an additional argument taking `fir.box_addr %var` value, though, this adds some redundancy to the argument list.

Alias analysis:

* For the purpose of alias analysis `fir.pack_array` should be considered a pass-through operation, meaning that when FIR alias analysis is looking for the source of a pointer, emboxed in the result box of `fir.pack_array`, the search is continued through the operation's argument box def-use.

#### fir.unpack_array operation

The operation has the following syntax:

```
fir.unpack_array %new_var to %var [stack ] [no_copy ] : !fir.box/class<!fir.array<...>>
```

The operation is either a no-op or deallocates the temporary array, and maybe copies the temporary array into the original array.

Arguments:

* `%new_var` is a value produced by `fir.pack_array`. In general, `%new_var` may become a block argument, and getting the other properties from the producer may not be possible. So some arguments are duplicated on `fir.unpack_array`.
* `stack` - indicates if `-fstack-arrays` is in effect for compiling this function.
* `no_copy` - indicates that, in case a temporary array is created, `%new_var` to `%var` copy is not required (`intent(in)` dummy argument case).

Memory effects are conservative, assuming that a copy and deallocation may happen:

* `MemFree` effect on either `AutomaticAllocationScopeResource` or `DefaultResource` depending on `stack` attribute.
* If there is no `no_copy`:
  * `MemWrite` effect on unknown value to indicate potential write into the original array. This effect should prevent hoisting any reads/writes of the original array above a `fir.unpack_array`, since those hoisted reads/writes may address the original array memory that has not been updated from the temporary copy yet.

### New attribute interface

The `TempCopyIsSafe` attribute interface provides means to generate programming model specific predicates saying whether repacking is safe or not at the point where it needs to be done. For example the OpenMP MLIR dialect may provide an attribute implementing this interface to generate a runtime check at the point of packing array `x` inside subroutine `repacking`. A conservative implementation might look like this:

```C
  repacking_is_safe = omp_get_num_team() == 1 && omp_get_num_threads() == 1;
```

The attribute interface will be used during lowering of `fir.pack_array` to generate the predicate and make the packing conditional and safe. This will allow applying repacking correctly in programs compiled with `-fopenmp`, and get the benefits of repacking in the serial parts of those programs.

Similarly, the OpenACC MLIR dialect may provide such an attribute to check if a device copy has been created for an array that is about to be packed, and prevent the repacking for cases like this:

```Fortran
program main
  real :: x(2,10)
  x = 0.0
  !$acc enter data copyin(x)
  do i=1,2
     call repacking(x(i,:))
  end do
  !$acc exit data copyout(x)
  print *, x
contains
  subroutine repacking(x)
    real :: x(:)
    !$acc parallel loop present(x)
    do j=1,10
       x(j) = x(j) + 1.0
    end do
  end subroutine repacking
end program main
```

Note that it may not be possible to detect all the cases where repacking breaks the original OpenACC/OpenMP program behavior. In the following case, it is unclear how to synchronize the repacking in both routines (especially, when the OpenACC directives are not lexically visible at the point of repacking):

```Fortran
program main
  real :: x(2,10)
  x = 0.0
  call repacking_in(x(1,:))
  call repacking_out(x(1,:))
  print *, x
contains
  subroutine repacking_in(x)
    real :: x(:)
    !$acc enter data copyin(x)
  end subroutine repacking_in
  subroutine repacking_out(x)
    real :: x(:)
    !$acc exit data copyout(x)
  end subroutine repacking_out
end program main
```

So the most conservative implementation of the predicate generator may be to always produce `false` value. Flang lowering may attach any number of such attributes to `fir.pack_array` depending on the compilation context and options.

[TBD] define `TempCopyIsSafe` attribute interface so that OpenACC/OpenMP dialects can provide their specific attributes, which can be used to generate static/runtime checks for safety of the temporary copy in particular context.

#### Alternatives/additions to the attribute interface

The following ideas were expressed during the review, and they are worth considering.

| |
| - |
| For OpenACC/OpenMP runtime to be able to detect/handle the presence of the original array and the temporary copy in the device data environment, descriptor flags/properties might be used to mark the copy's descriptor as such and provide a link to the original array (or its descriptor). It may be problematic to maintain such flags/properties, in general, because of the repacking that may happen, especially, in C code, where the flags/properties might be dropped. Moreover, a copy array might be repacked into another copy array multiple times, so a descriptor might need to keep a chain of associated arrays and it will have to be maintained as well.<br>An alternative to tracking the original-copy "association" might be compiler generated code notifying the OpenACC/OpenMP offload runtime about the copy being created/deleted for the original array, so that the offload runtime can disallow repacking or report an error when the repacking is definitely causing the program to behave incorrectly. The compiler may report the "association" to the runtime through callbacks provided by `TempCopyIsSafe` attribute interface, and this will require proper bookkeeping in the runtime specific to the array repacking. |
| There may be some uses for an API allowing to statically determine whether a given descriptor (SSA value) represent the repacked copy of the original array. For example, it may be in the form of an API in the OpenACC `MappableType` interface. This can be done with some limitations for the values produced by `fir.pack_array` that are dynamic (i.e. the copy is created conditionally based on the runtime checks). |
| The compiler can also try to statically determine the conditions where the array repacking might be unsafe, e.g. a presence of memory barriers or operations carrying implicit memory barriers, presence of atomic operations between the `pack/unpack` operations may indicate non-trivial handling of the array memory. Such checks may result in the removal of `pack/unpack` operations, and they can probably be done in a mandatory pass (not an optimization pass). At the same time, the result of the checks may depend on other optimization passes (e.g. inlining), so the behavior may be inconsistent between different optimization levels. |

### Lowering Fortran to FIR

Fortran code:

```Fortran
subroutine test(x,n)
  character(n) :: x(2:)
end subroutine test
```

HLFIR/FIR produced by Lowering:

```
  func.func @_QPtest(%arg0: !fir.box<!fir.array<?x!fir.char<1,?>>> {fir.bindc_name = "x"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) {
    %0 = fir.dummy_scope : !fir.dscope
    %1:2 = hlfir.declare %arg1 dummy_scope %0 {uniq_name = "_QFtestEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2 = fir.load %1#0 : !fir.ref<i32>
    %c0_i32 = arith.constant 0 : i32
    %3 = arith.cmpi sgt, %2, %c0_i32 : i32
    %4 = arith.select %3, %2, %c0_i32 : i32
    %c2_i64 = arith.constant 2 : i64
    %5 = fir.convert %c2_i64 : (i64) -> index
    %6 = fir.shift %5 : (index) -> !fir.shift<1>
    %repacked = fir.pack_array %arg0 typeparams %4 : !fir.box<!fir.array<?x!fir.char<1,?>>>
    %7:2 = hlfir.declare %repacked(%6) typeparams %4 dummy_scope %0 {uniq_name = "_QFtestEx"} : (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.shift<1>, i32, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.box<!fir.array<?x!fir.char<1,?>>>)
    fir.unpack_array %repacked to %arg0 : !fir.box<!fir.array<?x!fir.char<1,?>>>
    return
  }
```

#### Restrictions:

It is unsafe to create temporaries of assumed-shape dummy arrays that have `TARGET` attribute, because they can be accessed during the invocation of the subprograms not using direct reference of the dummy argument. Lowering must never produce the new operations for such dummy arguments. [TBD] a user option might be provided to override the default safe behavior.

The copy creation is also restricted for `ASYNCHRONOUS` and `VOLATILE` arguments. Such dummy arguments might be changed during the execution of their subprogram in an unpredictable manner, so creating a copy for them might be incorrect (Fortran 2023, Note 5 of section 15.5.2.5).

It does not make sense to generate the new operations for `CONTIGUOUS` arguments and for arguments with statically known element size that exceeds the `max-element-size` threshold.

#### Optional behavior

In case of the `whole` continuity mode or with 1-D array, Flang can propagate this information to `hlfir.declare` - this may improve optimizations down the road. This can be done iff the repacking has no dynamic constraints and/or heuristics. For example:

```
    %c0 = arith.constant 0 : index
    %6:3 = fir.box_dims %arg0, %c0 : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index) -> (index, index, index)
    %c2_i64 = arith.constant 2 : i64
    %7 = fir.convert %c2_i64 : (i64) -> index
    %8 = fir.shape_shift %7, %6#1 : (index, index) -> !fir.shapeshift<1>
    %repacked = fir.pack_array %arg0 typeparams %5 : !fir.box<!fir.array<?x!fir.char<1,?>>>
    %9 = fir.box_addr %repacked : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.ref<!fir.array<?x!fir.char<1,?>>>
    %10:2 = hlfir.declare %9(%8) typeparams %5 dummy_scope %0 {fortran_attrs = #fir.var_attrs<contiguous>, uniq_name = "_QFtestEx"} : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shapeshift<1>, i32, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.ref<!fir.array<?x!fir.char<1,?>>>)
```

This may complicate future `fir.pack_array` optimization passes, but it is worth considering.

### Lowering new operations

Lowering of the new operations (after all the optimizations) might be done in a FIR-to-FIR conversion pass.

`fir.pack_array` lowering might be done in the following steps:

* If there are dynamic constraints, generate a boolean `p1` that is set to true if repacking has to be done (depending on the constraints values and the original array descriptor). The IR might be cleaner if we generate a Fortran runtime call here.
* If there are attributes implementing `TempCopyIsSafe` attribute interface, then use the interface method to generate boolean predicates for each such attribute: `p2`, ..., `pn`.
  * [TBD] it seems that the runtime checks for the target offload programs will require a pure `PointerLike` value for the array start and the total byte size of the array (e.g. as `index` value).
* Compute `p = p1 && p2 && ... && pn`.
* Compute the total size of the temporary `required_size` (in elements).
* Compute the total size of the allocation `allocation_size = p ? required_size : 0`.
* If `stack` is present, then allocate memory on the stack `%space = fir.alloca <element-type>, %allocation_size`, otherwise, `%space = fir.allocmem !fir.array<?x<element-type>>, %allocation_size`.
* Embox `%space` into a temporary descriptor `%new_box`.
* `no_copy`:
  * If present, then `%new_box` is a new result of `fir.pack_array`.
  * If not present, then call Fortran runtime to do the copy into a pre-allocated temporary. Some part of the existing `PACK` runtime may be reused, but the intention is to have a shallow copy vs the deep copy implemented by `PACK`. The runtime call modifies `%new_box` and it becomes a new result of `fir.pack_array`.
  * For `OPTIONAL` absent argument, the result of the operation is the original absent box.

`fir.unpack_array` lowering:

* If `no_copy` is not present, then call Fortran runtime to do the the copy into the original array. The copy must be made iff `fir.box_addr %var != fir.box_addr %new_var`.

* If `stack` is not present, generate:

  ```
  %new_addr = fir.box_addr %new_var
  %orig_addr = fir.box_addr %orig_addr
  %cmp = arith.cmp neq, %new_addr, %orig_addr
  fir.if %cmp {
    fir.freemem %new_addr
  }
  ```
  Alternatively, we can let the Fortran runtime do the heap deallocation.

### Runtime

[TBD] define the runtime APIs.

### Optimization passes

[TBD] describe in more details optimization passes listed in [Overview](#overview). Define their place in the pipeline (e.g. with relation to FIR inlining).

#### Loop versioning

There is an existing optimization pass (controlled via `-f[no-]version-loops-for-stride`) that creates specialized versions of the loop nests which process assumed-shape dummy arrays. The pass adds a dynamic check for the array(s) contiguity (in the innermost dimension) and exposes the contiguity by rewriting the array(s) accesses via raw pointer references. This transformation may enable more unit-stride vectorization. The pass is beneficial only if the actual array argument is contiguous in the innermost dimension.

The array repacking is targeting better data cache utilization, and is not intended to enable more unit-strided vectorization for the assumed-shape arrays. At the same time, combining array repacking with the loop versioning may provide better performance for programs where the actual array arguments are non-contiguous, but then their repacked copies can be accessed using unit strides.

In cases where `fir.pack_array` is statically known to produce a copy that is contiguous in the innermost dimension, the loop versioning pass can skip the generation of the dynamic checks.

### Driver: user options

The following user options are proposed:

* `-frepack-arrays` - the option forces Flang to repack a non-contiguous assumed-shape dummy array into a temporary contiguous memory, which may result in faster accesses of the array. The compiler will insert special code in subprogram prologue to allocate a temporary array and copy the original array into the temporary; in subprogram epilogue, it will insert a copy from the temporary array into the original array and deallocate the temporary. The overhead of the allocation/deallocation and the copies may be significant depending on the array size. The compiler will try to optimize the unnecessary/unprofitable repacking.
* `-frepack-arrays-opts=[none|loop-only]` - the option enables optimizations that may eliminate the array repacking code depending on the array usage pattern:
  * `none` - no optimizations.
  * `loop-only` - the array repacking code will be removed in any subprogram where the array is not used inside a loop or an array expression.
* `-frepack-arrays-continuity=[whole|innermost]`:
  * `whole` - the option will repack arrays that are non-contiguous in any dimension (default).
  * `innermost` - the option will repack arrays that are non-contiguous in the innermost dimension.
* `-frepack-arrays-max-size=<int>` - arrays bigger than the specified size will not be repacked.
* `-frepack-arrays-max-element-size=<int>` - arrays with elements bigger than the specified size will not be repacked.
* `-frepack-arrays-min-stride=<int>` - arrays with the byte stride of the innermost dimension less than the specified value will not be repacked (e.g. it may not be profitable to repack arrays with the stride in the innermost dimension that is less than the data cache line size).
* [TBD] defaults for `-frepack-arrays-max-size`, `-frepack-arrays-max-element-size` and `-frepack-arrays-min-stride` will be defined based on benchmark tuning and might be target specific.

## Examples
### Example #1

```fortran
subroutine test(x,n,repeat,m1,m2,m3)
  real x(:)
  integer n,repeat,repeat_max
  integer i,j,idx1,idx2,idx3
  do j=1,repeat
     idx1=(j-1)/m1+0 ! always 0
     idx2=(j-1)/m2+n/3 ! always n/3
     idx3=(j-1)/m3+2*n/3 ! always 2*n/3
     do i=1,n/3
        x(idx1+i)=x(idx1+i)+1.0
        x(idx2+i)=x(idx2+i)+1.0
        x(idx3+i)=x(idx3+i)+1.0
     end do
  end do
end subroutine test

subroutine wrapper(x,n)
  interface
     subroutine test(x,n,repeat,m1,m2,m3)
       real x(:)
       integer n,repeat,m1,m2,m3
     end subroutine test
  end interface
  integer, parameter :: repeat=1000000
  integer n
  real x(n,n)
#if REPACKING
  real,allocatable :: tmp(:)
  allocate(tmp(SIZE(x,1)))
  tmp(:)=x(1,:)
  call test(tmp,n,repeat,repeat,repeat,repeat)
#else
  call test(x(1,:),n,repeat,repeat,repeat,repeat)
#endif
#if REPACKING
  x(1,:)=tmp(:)
  deallocate(tmp)
#endif
  print *,'X: ',x(1,::n/2)
end subroutine wrapper

program main
  interface
     subroutine wrapper(x,n)
       integer n
       real x(n,n)
     end subroutine wrapper
  end interface
  integer, parameter :: n=10000
  real x(n,n)
  x=0.0
  call wrapper(x,n)
end program main
```

### Example #2

```Fortran
program main
  use omp_lib
  real :: x(2,10)
  x = 0.0
  print *, x(1,:)
  !$omp parallel do num_threads(10) shared(x)
  do i=1,10
     call repacking(x(1,:), i)
  end do
  !$omp end parallel do
  print *, x(1,:)
contains
  subroutine repacking(x, i)
    real :: x(:)
    integer :: i
    print *, i !
    x(i) = x(i) + 1.0
  end subroutine repacking
end program main
```

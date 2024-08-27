<!--===- docs/OpenMP-declare-target.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM
   Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

# Introduction to Declare Target

In OpenMP `declare target` is a directive that can be applied to a function or
variable (primarily global) to notate to the compiler that it should be 
generated in a particular device's environment. In essence whether something 
should be emitted for host or device, or both. An example of its usage for 
both data and functions can be seen below.

```Fortran
module test_0
    integer :: sp = 0
!$omp declare target link(sp)
end module test_0

program main
    use test_0
!$omp target map(tofrom:sp)
    sp = 1
!$omp end target
end program
```

In the above example, we create a variable in a separate module, mark it 
as `declare target` and then map it, embedding it into the device IR and 
assigning to it. 


```Fortran
function func_t_device() result(i)
    !$omp declare target to(func_t_device) device_type(nohost)
        INTEGER :: I
        I = 1
end function func_t_device

program main
!$omp target
    call func_t_device()
!$omp end target
end program
```

In the above example, we are stating that a function is required on device
utilising `declare target`, and that we will not be utilising it on host, 
so we are in theory free to remove or ignore it there. A user could also 
in this case, leave off the `declare target` from the function and it 
would be implicitly marked `declare target any` (for both host and device), 
as it's been utilised within a target region.

# Declare Target as represented in the OpenMP Dialect

In the OpenMP Dialect `declare target` is not represented by a specific 
`operation`. Instead, it's an OpenMP dialect specific `attribute` that can be 
applied to any operation in any dialect, which helps to simplify the 
utilisation of it. Rather than replacing or modifying existing global or 
function `operations` in a dialect, it applies to it as extra metadata that
the lowering can use in different ways as is necessary. 

The `attribute` is composed of multiple fields representing the clauses you 
would find on the `declare target` directive i.e. device type (`nohost`, 
`any`, `host`) or the capture clause (`link` or `to`). A small example of 
`declare target` applied to a Fortran `real` can be found below:

```
fir.global internal @_QFEi {omp.declare_target = 
#omp.declaretarget<device_type = (any), capture_clause = (to)>} : f32 {
    %0 = fir.undefined f32
    fir.has_value %0 : f32
}
```

This would look similar for function style `operations`.

The application and access of this attribute is aided by an OpenMP Dialect 
MLIR Interface named `DeclareTargetInterface`, which can be utilised on 
operations to access the appropriate interface functions, e.g.:

```C++
auto declareTargetGlobal = 
llvm::dyn_cast<mlir::omp::DeclareTargetInterface>(Op.getOperation());
declareTargetGlobal.isDeclareTarget();
```

# Declare Target Fortran OpenMP Lowering

The initial lowering of `declare target` to MLIR for both use-cases is done
inside of the usual OpenMP lowering in flang/lib/Lower/OpenMP.cpp. However, 
some direct calls to `declare target` related functions from Flang's 
lowering bridge in flang/lib/Lower/Bridge.cpp are made.

The marking of operations with the declare target attribute happens in two 
phases, the second one optional and contingent on the first failing. The 
initial phase happens when the declare target directive and its clauses 
are initially processed, with the primary data gathering for the directive and
clause happening in a function called `getDeclareTargetInfo`. This is then used
to feed the `markDeclareTarget` function, which does the actual marking 
utilising the `DeclareTargetInterface`. If it encounters a variable or function
that has been marked twice over multiple directives with two differing device
types (e.g. `host`, `nohost`), then it will swap the device type to `any`.

Whenever we invoke `genFIR` on an `OpenMPDeclarativeConstruct` from the 
lowering bridge, we are also invoking another function called 
`gatherOpenMPDeferredDeclareTargets`, which gathers information relevant to the
application of the `declare target` attribute. This information 
includes the symbol that it should be applied to, device type clause, 
and capture clause, and it is stored in a vector that is part of the lowering
bridge's instantiation of the `AbstractConverter`. It is only stored if we 
encounter a function or variable symbol that does not have an operation 
instantiated for it yet. This cannot happen as part of the 
initial marking as we must store this data in the lowering bridge and we 
only have access to the abstract version of the converter via the OpenMP 
lowering. 

The information produced by the first phase is used in the second phase, 
which is a form of deferred processing of the `declare target` marked 
operations that have delayed generation and cannot be proccessed in the 
first phase. The main notable case this occurs currently is when a 
Fortran function interface has been marked. This is 
done via the function 
`markOpenMPDeferredDeclareTargetFunctions`, which is called from the lowering
bridge at the end of the lowering process allowing us to mark those where 
possible. It iterates over the data previously gathered by 
`gatherOpenMPDeferredDeclareTargets` 
checking if any of the recorded symbols have now had their corresponding 
operations instantiated and applying the declare target attribute where 
possible utilising `markDeclareTarget`. However, it must be noted that it 
is still possible for operations not to be generated for certain symbols, 
in particular the case of function interfaces that are not directly used 
or defined within the current module. This means we cannot emit errors in 
the case of left-over unmarked symbols. These must (and should) be caught 
by the initial semantic analysis.

NOTE: `declare target` can be applied to implicit `SAVE` attributed variables.
However, by default Flang does not represent these as `GlobalOp`'s, which means
we cannot tag and lower them as `declare target` normally. Instead, similarly
to the way `threadprivate` handles these cases, we raise and initialize the 
variable as an internal `GlobalOp` and apply the attribute. This occurs in the
flang/lib/Lower/OpenMP.cpp function `genDeclareTargetIntGlobal`.

# Declare Target Transformation Passes for Flang

There are currently two passes within Flang that are related to the processing 
of `declare target`:
* `MarkDeclareTarget` - This pass is in charge of marking functions captured
(called from) in `target` regions or other `declare target` marked functions as
`declare target`. It does so recursively, i.e. nested calls will also be 
implicitly marked. It currently will try to mark things as conservatively as 
possible, e.g. if captured in a `target` region it will apply `nohost`, unless
it encounters a `host` `declare target` in which case it will apply the `any` 
device type. Functions are handled similarly, except we utilise the parent's 
device type where possible.
* `FunctionFiltering` - This is executed after the `MarkDeclareTarget`
pass, and its job is to conservatively remove host functions from
the module where possible when compiling for the device. This helps make 
sure that most incompatible code for the host is not lowered for the 
device. Host functions with `target` regions in them need to be preserved 
(e.g. for lowering the `target region`(s) inside). Otherwise, it removes 
any function marked as a `declare target host` function and any uses will be 
replaced with `undef`'s so that  the remaining host code doesn't become broken. 
Host functions with `target` regions are marked with a `declare target host` 
attribute so they will be removed after outlining the target regions contained
inside.

While this infrastructure could be generally applicable to more than just Flang, 
it is only utilised in the Flang frontend, so it resides there rather than in 
the OpenMP dialect codebase. 

# Declare Target OpenMP Dialect To LLVM-IR Lowering

The OpenMP dialect lowering of `declare target` is done through the 
`amendOperation` flow, as it's not an `operation` but rather an 
`attribute`. This is triggered immediately after the corresponding
operation has been lowered to LLVM-IR. As it is applicable to
different types of operations, we must specialise this function for 
each operation type that we may encounter. Currently, this is 
`GlobalOp`'s and `FuncOp`'s.

`FuncOp` processing is fairly simple. When compiling for the device, 
`host` marked functions are removed, including those that could not 
be removed earlier due to having `target` directives within. This 
leaves `any`, `device` or indeterminable functions left in the 
module to lower further. When compiling for the host, no filtering is 
done because `nohost` functions must be available as a fallback 
implementation.

For `GlobalOp`'s, the processing is a little more complex. We 
currently leverage the `registerTargetGlobalVariable` and 
`getAddrOfDeclareTargetVar` `OMPIRBuilder` functions shared with Clang. 
These two functions invoke each other depending on the clauses and options 
provided to the `OMPIRBuilder` (in particular, unified shared memory). Their
main purposes are the generation of a new global device pointer with a 
"ref_" prefix on the device and enqueuing metadata generation by the 
`OMPIRBuilder` to be produced at module finalization time. This is done 
for both host and device and it links the newly generated device global 
pointer and the host pointer together across the two modules.

Similarly to other metadata (e.g. for `TargetOp`) that is shared across
both host and device modules, processing of `GlobalOp`'s in the device 
needs access to the previously generated host IR file, which is done 
through another `attribute` applied to the `ModuleOp` by the compiler 
frontend. The file is loaded in and consumed by the `OMPIRBuilder` to 
populate it's `OffloadInfoManager` data structures, keeping host and 
device appropriately synchronised.

The second (and more important to remember) is that as we effectively replace
the original LLVM-IR generated for the `declare target` marked `GlobalOp` we
have some corrections we need to do for `TargetOp`'s (or other region 
operations that use them directly) which still refer to the original lowered
global operation. This is done via `handleDeclareTargetMapVar` which is invoked
as the final function and alteration to the lowered `target` region, it's only
invoked for device as it's only required in the case where we have emitted the
"ref" pointer , and it effectively replaces all uses of the originally lowered
global symbol, with our new global ref pointer's symbol. Currently we do not
remove or delete the old symbol, this is due to the fact that the same symbol
can be utilised across multiple target regions, if we remove it, we risk 
breaking lowerings of target regions that will be processed at a later time. 
To appropriately delete these no longer necessary symbols we would need a 
deferred removal process at the end of the module, which is currently not in 
place. It may be possible to store this information in the OMPIRBuilder and 
then perform this cleanup process on finalization, but this is open for 
discussion and implementation still.

# Current Support

For the moment, `declare target` should work for:
* Marking functions/subroutines and function/subroutine interfaces for 
generation on host, device or both.
* Implicit function/subroutine capture for calls emitted in a `target` region 
or explicitly marked `declare target` function/subroutine. Note: Calls made 
via arguments passed to other functions must still be themselves marked 
`declare target`, e.g. passing a `C` function pointer and invoking it, then 
the interface and the `C` function in the other module must be marked 
`declare target`, with the same type of marking as indicated by the 
specification.
* Marking global variables with `declare target`'s `link` clause and mapping 
the data to the device data environment utilising `declare target`. This may 
not work for all types yet, but for scalars and arrays of scalars, it 
should.

Doesn't work for, or needs further testing for:
* Marking the following types with `declare target link` (needs further 
testing):
    * Descriptor based types, e.g. pointers/allocatables.
    * Derived types.
    * Members of derived types (use-case needs legality checking with OpenMP
specification).
* Marking global variables with `declare target`'s `to` clause. A lot of the
lowering should exist, but it needs further testing and likely some further 
changes to fully function.

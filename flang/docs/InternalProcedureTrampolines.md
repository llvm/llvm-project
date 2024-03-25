<!--===- docs/InternalProcedureTrampolines.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

# Trampolines for pointers to internal procedures.

## Overview

```fortran
subroutine host()
  integer :: local
  local = 10
  call internal()
  return

  contains
  subroutine internal()
    print *, local
  end subroutine internal
end subroutine host
```

Procedure code generated for subprogram `internal()` must have access to the scope of
its host procedure, e.g. to access `local` variable. Flang achieves this by passing
an extra argument to `internal()` that is a tuple of references to all variables
used via host association inside `internal()`. We will call this extra argument
a static chain link.

Fortran standard 2008 allowed using internal procedures as actual arguments for
procedure pointer targets:

> Fortran 2008 contains several extensions to Fortran 2003; some of these are listed below.
>
> * An internal procedure can be used as an actual argument or procedure pointer target.
>
> NOTE 12.18
>
> An internal procedure cannot be invoked using a procedure pointer from either Fortran or C after the host instance completes execution, because the pointer is then undefined. While the host instance is active, however, the internal procedure may be invoked from outside of the host procedure scoping unit if that internal procedure was passed as an actual argument or is the target of a procedure pointer.

Special handling is required for the internal procedures that might be invoked
via an argument association or via pointer.
This document describes Flang implementation to support it.

> NOTE: in some languages/extensions the static chain may contain links
to more than one stack frame, while Fortra's static chain only ever
has a link to a single host procedure.

## Flang current implementation

### Examples

Internal procedure as procedure pointer target:

```fortran
module other
  abstract interface
     function callback()
       integer :: callback
     end function callback
  end interface
  contains
  subroutine foo(fptr)
    procedure(callback), pointer :: fptr
    ! `fptr` is pointing to `callee`, which needs the static chain link.
    print *, fptr()
  end subroutine foo
end module other

subroutine host(local)
  use other
  integer :: local
  procedure(callback), pointer :: fptr
  fptr => callee
  call foo(fptr)
  return

  contains

  function callee()
    integer :: callee
    callee = local
  end function callee
end subroutine host

program main
  call host(10)
end program main
```

Internal procedure as actual argument (F90 style):

```fortran
module other
  contains
  subroutine foo(fptr)
    interface
      integer function fptr()
      end function
    end interface
    ! `fptr` is pointing to `callee`, which needs the static chain link.
    print *, fptr()
  end subroutine foo
end module other

subroutine host(local)
  use other
  integer :: local
  call foo(callee)
  return

  contains

  function callee()
    integer :: callee
    callee = local
  end function callee
end subroutine host

program main
  call host(10)
end program main
```

Internal procedure as actual argument (F77 style):

```fortran
module other
  contains
  subroutine foo(fptr)
    integer :: fptr
    ! `fptr` is pointing to `callee`, which needs the static chain link.
    print *, fptr()
  end subroutine foo
end module other

subroutine host(local)
  use other
  integer :: local
  call foo(callee)
  return

  contains

  function callee()
    integer :: callee
    callee = local
  end function callee
end subroutine host

program main
  call host(10)
end program main
```

In all cases, the call sequence implementing `fptr()` call site inside `foo()`
must pass the stack chain link to the actual function `callee()`.

### Usage of trampolines in Flang

`BoxedProcedure` pass recognizes `fir.emboxproc` operations that
embox a subroutine address together with the static chain link,
and transforms them into a sequence of operations that replace
the result of `fir.emboxproc` with an address of a trampoline.
Eventually, it is the address of the trampoline that is passed
as an actual argument to `foo()`.

The trampoline has the following structure:

```asm
callee_trampoline:
  MOV static-chain-address, R#
  JMP callee-address
```

Where:
- `callee-address` is the address of function `callee()`.
- `static-chain-address` - the address of the static chain
  object created inside `host()`.
- `R#` is a target specific register.

In MLIR LLVM dialect the replacement looks like this:

```
    llvm.call @llvm.init.trampoline(%8, %9, %7) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> ()
    %10 = llvm.call @llvm.adjust.trampoline(%8) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %11 = llvm.bitcast %10 : !llvm.ptr<i8> to !llvm.ptr<func<void ()>>
    llvm.call @_QMotherPfoo(%11) {fastmathFlags = #llvm.fastmath<fast>} : (!llvm.ptr<func<void ()>>) -> ()

```

So any call of `fptr` inside `foo()` will result in invocation of the trampoline.
The trampoline will setup `R#` register and jump to `callee()` directly.

The ABI of `callee()` is adjusted using `llvm.nest` call argument attribute,
so that the target code generator assumes the static chain argument is passed
to `callee()` in `R#`:

```
  llvm.func @_QFhostPcallee(%arg0: !llvm.ptr<struct<(ptr<i32>)>> {fir.host_assoc, llvm.nest}) -> i32 attributes {fir.internal_proc} {
```

#### Trampoline handling

Currently used [llvm.init.trampoline intrinsic](https://llvm.org/docs/LangRef.html#trampoline-intrinsics)
expects that the memory for the trampoline content is passed to it as the first argument.
The memory has to be writeable at the point of the intrinsic call, and it has to be executable
at any point where `callee()` might be ivoked via the trampoline.

`@llvm.init.trampoline` intrinsic initializes the trampoline area in a target-specific manner
so that being executed: the trampoline sets a target-specific register to be equal to the third argument
(which is a static chain address), and then calls the function defined by the second argument.

Some targets may perform additional actions to guarantee the readiness of the trampoline for execution,
e.g. [call](https://github.com/llvm/llvm-project/blob/main/compiler-rt/lib/builtins/trampoline_setup.c)
`__clear_cache` or do something else.

For each internal procedure a trampoline may be initialized once per the host invocation.

The target-specific address of the new trampoline function must be taken via another intrinsic call:

```
%p = call i8* @llvm.adjust.trampoline(i8* %trampoline_address)
```

Note that value of `%p` is equal to `%tramp1` in most cases, but this is not
a requirement - this is partly [why](https://lists.llvm.org/pipermail/llvm-dev/2011-August/042845.html)
the second intrinsic was introduced:

> ```
> By the way an example of adjust_trampoline is ARM, which or's a 1 into the address of the trampoline.  When the pointer is called the processor sees the 1 and puts itself into thumb mode.

Currently, the trampolines are allocated on the stack of `host()` subroutine,
so that they are available throughout the life span of `host()` and are
automatically deallocated at the end of `host()` invocation.
Unfortunately, this requires the program stack to be writeable and executable
at the same time, which might be a security concern.

> NOTE: LLVM's AArch64 backend supports `nest` attribute, but it does not seem to support trampoline intrinsics.

## Alternative implementation(s)

To address the security risk we may consider managing the trampoline memory
in a way that it is not writeable and executable at the same time.
One of the options is to use separate allocations for the trampoline code
and the trampoline "data".

The trampolines may be located in non-writeable executable memory:
```asm
trampoline0:
  MOV (TDATA[0].static_chain_address), R#
  JMP (TDATA[0].callee_address)
trampoline1:
  MOV (TDATA[1].static_chain_address), R#
  JMP (TDATA[1].callee_address)
...
```

The `TDATA` memory is writeable and contains *<static chain address, function address>*
for each of the trampolines.

A runtime support library may provide APIs for initializing/accessing/deallocating
the trampolines that can be used by `BoxedProcedure` pass.

### Implementation considerations

* The static chain address still has to be passed in fixed target-specific register,
  and the implementations that rely on LLVM back-ends can use `nest` attribute for this.

* The trampoline area must be able to grow, because there can be a trampoline
  for each internal procedure per host invocation, and an internal procedure can call
  the host recursively. This means that the amount of trampolines in one thread
  may grow pretty quickly.

  ```fortran
  recursive subroutine host(local)
    use other
    integer :: local
    call foo(callee)
    return

    contains

    function callee()
      integer :: callee
      if (local .le. CONST_N) then
         call host(local + 1)
      endif
    end function callee
  end subroutine host
  ```

* On the other hand, putting a hard limit on the number of trampolines live at the same time
  allows putting the trampolines into the static code segment.

* Each thread may have its own dynamic trampoline area to reduce the number
  of required locks.

* Some support is required for the offload devices.

* Each trampoline invocation implies two indirect accesses with this approach.

### Fortran runtime support

The following APIs are suggested:

```c++
/**
 * \brief Initializes new trampoline and returns its internal handle.
 *
 * Initializes new trampoline with the given \p callee_address
 * and \p static_chain_address, and returns the new trampoline's
 * internal handle. The compiler calls this method once per host
 * invocation for each internal procedure that will need its address
 * passed around.
 *
 * The initialization is reserving a new entry in TDATA and
 * initializes the entry with the given \p callee_address and
 * \p static_chain_address; it is also reserving a new entry
 * in the trampoline area that is using the corresponding TDATA entry.
 *
 * Optional:
 *   \p scratch may be used to switch between the trampoline pool
 *   and llvm.init.trampoline implementation, e.g. if compiler passes
 *   non-null \p scratch it will be used as a writeable/executable
 *   memory for the new trampoline.
 */
const void *InitTrampoline([[maybe_unused]] void *scratch,
                           const void *callee_address,
                           const void *static_chain_address);

/**
 * \brief Returns the trampoline's address for the given handle.
 *
 * \p handle is a value returned by InitTrampoline().
 * The result of AdjustTrampoline() is the actual callable
 * trampoline's address.
 *
 * Optional: may be implemented via llvm.adjust.trampoline.
 */
const void *AdjustTrampoline(const void *handle);

/**
 * \brief Frees internal resources occupied for the given trampoline.
 *
 * The compiler must call this API at every exit from the host function.
 *
 * Optional: may be no-op, if LLVM trampolines are used underneath.
 */
void FreeTrampoline(void *handle);
```

`InitTrampoline` will do the initial allocation of the TDATA memory
and the trampoline area followed by the initialization of the trampoline
area with the binary code to "link" the trampolines with the corresponding
TDATA entries. After the initial allocation the trampoline area is made
executable and not writeable.

If there is an available entry in the TDATA/trampoline area, then the function
will initialized the TDATA entry with the given arguments and return
a handle to the trampoline entry.

`FreeTrampoline` will free the reserved entry.

> NOTE: `FreeTrampoline` may reset the `callee_address` in the trampoline
being freed to a runtime library function that complains about a dead
internal procedure being called. This provides some runtime diagnostics
of dangling procedure pointer usage. Such freed trampolines may still
have to be reclaimed, if new trampoline is requested and the trampoline
area is all used.

#### Sample IR

```
    // Init the trampoline once per host procedure invocation
    // (i.e. when the procedure address is emboxed).
    %handle = llvm.call @_FortranAInitTrampoline(%nullptr, %9, %7) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.ptr<i8>
    // Get the actual internal procedure address once per host procedure invocation.
    %10 = llvm.call @_FortranAAdjustTrampoline(%handle) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %11 = llvm.bitcast %10 : !llvm.ptr<i8> to !llvm.ptr<func<void ()>>
    llvm.call @_QMotherPfoo(%11) {fastmathFlags = #llvm.fastmath<fast>} : (!llvm.ptr<func<void ()>>) -> ()
    // The trampoline deallocation must be done only at the exits from the host procedure.
    llvm.call @_FortranAFreeTrampoline(%handle) : (!llvm.ptr<i8>) -> ()
```

### Implementation options

We may try to reuse [libffi](https://github.com/libffi/libffi) implementation for __static trampolines__:
* Initial implementation added support for x64, i386, aarch64 and arm on Linux: https://github.com/libffi/libffi/pull/624
* Follow-up patches:
  * Added support for Cygwin: https://github.com/libffi/libffi/commit/a1130f37712c03957c9b0adf316cd006fa92a60b
  * Added support for LoongArch: https://github.com/libffi/libffi/pull/723
  * Page protection for iOS devices: https://github.com/libffi/libffi/pull/718
  * Fix for trampoline code for x32: https://github.com/libffi/libffi/pull/657
* The author (@madvenka786) initially [proposed](https://sourceware.org/pipermail/libffi-discuss/2021/002587.html) to make the trampoline APIs public,
  but this was not a requirement at the time and the APIs were made private.
  If we want to rely on `libffi`, the APIs have to be made public.
* We may also try to extract the static trampolines implementation from `libffi`
  into separate library (e.g. `libstatictramp` as mentioned [here](https://sourceware.org/pipermail/libffi-discuss/2021/002592.html)).

Flang's own implementation for trampolines have an advantage that,
having to support the only Fortran/C interoperable calling convention,
the implementation may reduce the trampoline overhead. For example,
it may avoid saving/restoring the scratch registers used by the trampoline code,
and just clobber some of them according to the particular ABI.

At this point, the recommended approach is to implement the trampoline
support in Flang runtime.

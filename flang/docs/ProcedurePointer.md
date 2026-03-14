<!--===- docs/ProcedurePointer.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

# Procedure Pointer

A procedure pointer is a procedure that has the EXTERNAL and POINTER attributes.

This document summarizes what of context the procedure pointers should appear,
and how they are lowered to FIR.

The current plan is to use/extend the `BoxedProcedure` pass for the conversion
to LLVM IR, and thus will not be lowering the procedure-pointer-related
operations to LLVM IR in `CodeGen.cpp`.

## Fortran standard

Here is a list of the sections and constraints of the Fortran standard involved
for procedure pointers.

- 8.5.4 Components
  - C757
  - C758
  - C759
- 8.5.9: EXTERNAL attribute
- 8.5.14: POINTER attribute
  - C853
  - A procedure pointer shall not be referenced unless it is pointer associated
    with a target procedure.
- 8.5.15 PROTECTED attribute
  - C855
- 8.5.16 SAVE attribute
  - (4) A procedure pointer declared in the scoping unit of a main program,
        module, or submodule implicitly has the SAVE attribute.
- 8.10.2.1 COMMON statement
  - C8119
- 10.2.2.2 Pointer assignment statement
  - C1028
  - C1029
- 10.2.2.4 Procedure pointer assignment
- 11.1.3 ASSOCIATE construct
  - C1005
- 12.6.3 Data transfer input/output list
  - C1233
- 15.2.2.4 Procedure pointers
  - A procedure pointer may be pointer associated with an external procedure, an
    internal procedure, an intrinsic procedure, a module procedure, or a dummy
    procedure that is not a procedure pointer.
- 15.4.3.6 Procedure declaration statement
- 15.5.2.9(5) Actual arguments associated with dummy procedure entities
- 16.9.16 ASSOCIATED(POINTER [, TARGET])
  - POINTER may be a procedure pointer, and TARGET may be proc-target in a
    pointer assignment statement (10.2.2).
- 16.9.144 NULL([MOLD])
  - MOLD may be a procedure pointer.
- 18.2.3.4 C_F_PROCPOINTER(CPTR, FPTR)
  - FPTR shall be a procedure pointer, and not be a component of a coindexed
    object.
- C.1.1 A procedure that is not a procedure pointer can be an actual argument
  that corresponds to a procedure pointer dummy argument with the INTENT(IN)
  attribute.

---

## Representation in FIR

### Procedure pointer `!fir.ref<!fir.boxproc<T>>`

A procedure pointer may have an explicit or implicit interface. T in
`!fir.ref<!fir.boxproc<T>>` is the function type, which is `() -> ()` if the
procedure pointer has the implicit interface declared as
`procedure(), pointer :: p`.

A procedure declaration statement specifies EXTERNAL attribute (8.5.9) for all
entities for all entities in the procedure declaration list.

### Actual arguments associated with dummy procedure entities

The actual argument may be a procedure pointer, a valid target for the dummy
pointer, a reference to the NULL() intrinsic, or a reference to a function that
returns a procedure pointer.

If the interface is explicit, and the dummy argument is procedure pointer, the
reference is resolved as the pointer to the procedure; otherwise, the reference
is resolved as the pointer target.

**Fortran case 1**
```fortran
subroutine proc_pointer_dummy_argument(p)
  interface
    function func(x)
      integer :: x
    end function func
  end interface
  procedure(func), pointer :: p
  call foo1(p)
  call foo2(p)
contains
  subroutine foo2(q)
    interface
      function func(x)
        integer :: x
      end function func
    end interface
    procedure(func), pointer :: q
  end subroutine foo2
end subroutine proc_pointer_dummy_argument
```

**FIR for case 1**
```
func.func private @foo1(!fir.boxproc<(!fir.ref<i32>) -> f32>)
func.func private @foo2(!fir.ref<!fir.boxproc<(!fir.ref<i32>) -> f32>>)

func.func @proc_pointer_dummy_argument(%0 : !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> f32>>) {
  %1 = fir.load %0 : !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> f32>>
  fir.call @foo1(%1) : (!fir.boxproc<(!fir.ref<i32>) -> f32>) -> ()
  fir.call @foo2(%0) : (!fir.ref<!fir.boxproc<(!fir.ref<i32>) -> f32>>) -> ()
  return
}
```

**Fortran case 2**
```fortran
subroutine proc_pointer_global()
  interface
    function func(x)
      integer :: x
    end function func
  end interface
  procedure(func), pointer, save :: p
  call foo1(p)
  call foo2(p)
contains
  subroutine foo2(q)
    interface
      function func(x)
        integer :: x
      end function func
    end interface
    procedure(func), pointer :: q
  end subroutine foo2
end subroutine proc_pointer_global
```

**FIR for case 2**
```
func.func private @foo1(!fir.boxproc<(!fir.ref<i32>) -> f32>)
func.func private @foo2(!fir.ref<!fir.boxproc<(!fir.ref<i32>) -> f32>>)

fir.global internal @ProcedurePointer : !fir.boxproc<(!fir.ref<i32>) -> f32> {
  %0 = fir.zero_bits (!fir.ref<i32>) -> f32
  %1 = fir.emboxproc %0 : ((!fir.ref<i32>) -> f32) -> !fir.boxproc<(!fir.ref<i32>) -> f32>
  fir.has_value %1 : !fir.boxproc<(!fir.ref<i32>) -> f32>
}

func.func @proc_pointer_global() {
  %0 = fir.address_of(@ProcedurePointer) : !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> f32>>
  %1 = fir.load %0 : !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> f32>>
  fir.call @foo1(%1) : (!fir.boxproc<(!fir.ref<i32>) -> f32>) -> ()
  fir.call @foo2(%0) : (!fir.ref<!fir.boxproc<(!fir.ref<i32>) -> f32>>) -> ()
  return
}
```

**Fortran case 3**
```fortran
subroutine proc_pointer_local()
  interface
    function func(x)
      integer :: x
    end function func
  end interface
  procedure(func), pointer :: p
  call foo1(p)
  call foo2(p)
contains
  subroutine foo2(q)
    interface
      function func(x)
        integer :: x
      end function func
    end interface
    procedure(func), pointer :: q
  end subroutine foo2
end subroutine proc_pointer_local
```

**FIR for case 3**
```
func.func private @foo1(!fir.boxproc<(!fir.ref<i32>) -> f32>)
func.func private @foo2(!fir.ref<!fir.boxproc<(!fir.ref<i32>) -> f32>>)

func.func @proc_pointer_local() {
  %0 = fir.alloca !fir.boxproc<(!fir.ref<i32>) -> f32>
  %1 = fir.zero_bits (!fir.ref<i32>) -> f32
  %2 = fir.emboxproc %1 : ((!fir.ref<i32>) -> f32) -> !fir.boxproc<(!fir.ref<i32>) -> f32>
  fir.store %2 to %0 : !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> f32>>
  %4 = fir.load %0 : !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> f32>>
  fir.call @foo1(%4) : (!fir.boxproc<(!fir.ref<i32>) -> f32>) -> ()
  fir.call @foo2(%0) : (!fir.ref<!fir.boxproc<(!fir.ref<i32>) -> f32>>) -> ()
  return
}
```

It is possible to pass procedure pointers to a C function. If the C function has
an explicit interface in fortran code, and the dummy argument is a procedure
pointer, the code passes a pointer to the procedure as the actual argument
(see Case 5); Otherwise, the code passes the procedure pointer target as the
actual argument (see Case 4).

**Case 4**
```c
void func_(void (*foo)(int *)) {
  int *x, y = 1;
  x = &y;
  foo(x);
}
```
```fortran
program main
  procedure(), pointer :: pp
  pp=>print_x
  call func(pp)
contains
  subroutine print_x(x)
    integer :: x
    print *, x
  end
end
```

Note that the internal procedure is not one good usage, but it works in
implementation. It is better to use BIND(C) external or module procedure as
right-hand side proc-target.

**Case 5**
```c
void func_(void (**foo)(int *)) {
  int *x, y = 1;
  x = &y;
  (*foo)(x);
}
```
```fortran
program main
  interface
    subroutine func(p)
      procedure(), pointer :: p
    end
  end interface
  procedure(), pointer :: pp
  pp=>print_x
  call func(pp)
contains
  subroutine print_x(x)
    integer :: x
    print *, x
  end
end
```

Case 4 and Case 5 are not recommended from Fortran 2003 standard, which provides
the feature of interoperability with C to handle this. Specifically,
C_F_PROCPOINTER is used to associate a procedure pointer with the target of a C
function pointer. C_FUNPTR is also designed for interoperability with any C
function pointer type.

### Procedure pointer to function returning a character type

The dummy procedure pointer may not have a function type with an assumed length
due to C721 and C723.

### Procedure pointer to internal procedure

Initially the current plan is to implement pointers to internal procedures
using the LLVM Trampoline intrinsics. This has the drawback of requiring the
stack to be executable, which is a security hole. To avoid this, we will need
[improve the implementation](InternalProcedureTrampolines.md) to use heap-resident thunks.

### Procedure pointer assignment `p => proc`

The right-hand side may be a procedure, a procedure pointer, or a function whose
result is a procedure pointer.

The procedure could be a BIND(C) procedure. The lowering of it is the same as
that of an external or module procedure. The case of internal procedure has been
discussed above.

```c
#include<stdio.h>
void func_(int *x) {
  printf("%d\n", *x);
}
```
```fortran
program main
  interface
    subroutine func(x) bind(C)
      integer :: x
    end
  end interface
  procedure(func), bind(C, name="func_") :: proc
  procedure(func), pointer :: pp
  integer :: x = 5
  pp=>proc
  call pp(x)
end
```

**Fortran case**
```fortran
subroutine proc_pointer_assignment(arg0, arg1)
  interface
    function func(x)
      integer :: x
    end
  end interface
  procedure(func), pointer :: arg0, arg1
  real, external, bind(C, name="Procedure") :: proc
  arg0=>proc    ! case 1
  arg0=>arg1    ! case 2
  arg0=>reffunc ! case 3
contains
  function reffunc() result(pp)
    interface
      function func(x)
        integer :: x
      end
    end interface
    procedure(func), pointer :: pp
  end
end
function proc(x) bind(C, name="Procedure")
  integer :: x
  proc = real(x)
end
```

**FIR**
```
func.func @Procedure(%arg0 : !fir.ref<i32>) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "res", uniq_name = "_QFfuncEres"}
  %1 = fir.load %arg0 : !fir.ref<i32>
  %2 = fir.convert %1 : (i32) -> f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}

func.func @Reference2Function() -> !fir.boxproc<(!fir.ref<i32>) -> f32> {
  %0 = fir.alloca !fir.boxproc<(!fir.ref<i32>) -> f32>
  %1 = fir.load %0 : !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> f32>>
  return %1 : !fir.boxproc<(!fir.ref<i32>) -> f32>
}

func.func @proc_pointer_assignment(%arg0 : !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> f32>>, %arg1 : !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> f32>>) {
  %0 = fir.alloca !fir.boxproc<(!fir.ref<i32>) -> f32> {bindc_name = ".result"}
  // case 1: assignment from external procedure
  %1 = fir.address_of(@Procedure) : (!fir.ref<i32>) -> f32
  %2 = fir.emboxproc %1 : ((!fir.ref<i32>) -> f32) -> !fir.boxproc<(!fir.ref<i32>) -> f32>
  fir.store %2 to %arg0 : !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> f32>>
  // case2: assignment from procdure pointer
  %3 = fir.load %arg1 : !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> f32>>
  fir.store %3 to %arg0 : !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> f32>>
  // case3: assignment from a reference to a function whose result is a procedure pointer
  %4 = fir.call @Reference2Function() : () -> !fir.boxproc<(!fir.ref<i32>) -> f32>
  fir.store %4 to %0 : !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> f32>>
  %5 = fir.load %0 : !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> f32>>
  fir.store %5 to %arg0 : !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> f32>>
  return
}
```

### Procedure pointer components

Having procedure pointers in derived types permits `methods` to be dynamically
bound to objects. Such procedure pointer components will have the type
!fir.boxproc<T>.

**Fortran**
```fortran
subroutine proc_pointer_component(a, i, f)
  interface
    function func(x)
      integer :: x
    end
  end interface
  type matrix
    real :: element(2,2)
    procedure(func), pointer, nopass :: solve
  end type
  integer :: i
  procedure(func) :: f
  type(matrix) :: a
  a%solve=>f
  r = a%solve(i)
end subroutine proc_pointer_component
```

**FIR**
```
func.func @proc_pointer_component(%arg0 : !fir.boxproc<(!fir.ref<i32>) -> f32>, %arg1: !fir.ref<i32>) {
  %0 = fir.alloca !fir.type<_QFtestTmatrix{element:!fir.array<2x2xf32>,solve:!fir.boxproc<() -> ()>}>
  %1 = fir.field_index solve, !fir.type<_QFtestTmatrix{element:!fir.array<2x2xf32>,solve:!fir.boxproc<() -> ()>}>
  %2 = fir.coordinate_of %0, %1 : (!fir.ref<!fir.type<_QFtestTmatrix{element:!fir.array<2x2xf32>,solve:!fir.boxproc<() -> ()>}>>, !fir.field) -> !fir.ref<!fir.boxproc<() -> ()>>
  %3 = fir.convert %arg0 : (!fir.boxproc<(!fir.ref<i32>) -> f32>) ->  !fir.boxproc<() -> ()>
  fir.store %3 to %2 : !fir.ref<!fir.boxproc<() -> ()>>
  %4 = fir.field_index solve, !fir.type<_QFtestTmatrix{element:!fir.array<2x2xf32>,solve:!fir.boxproc<() -> ()>}>
  %5 = fir.coordinate_of %0, %4 : (!fir.ref<!fir.type<_QFtestTmatrix{element:!fir.array<2x2xf32>,solve:!fir.boxproc<() -> ()>}>>, !fir.field) -> !fir.ref<!fir.boxproc<() -> ()>>
  %6 = fir.load %5 : !fir.ref<!fir.boxproc<() -> ()>>
  %7 = fir.convert %6 : (!fir.boxproc<() -> ()>) -> !fir.boxproc<(!fir.ref<i32>) -> f32>
  %8 = fir.box_addr %7 : (!fir.boxproc<(!fir.ref<i32>) -> f32>) -> ((!fir.ref<i32>) -> f32)
  %9 = fir.call %8(%arg1) : (!fir.ref<i32>) -> f32
  return
}
```

---

## Testing

The lowering part is tested with LIT tests in tree, but the execution tests are
useful for full testing.

LLVM IR testing is also helpful with the initial check. A C function pointer is
semantically equivalent to a Fortran procedure in LLVM IR level, and a pointer
to a C function pointer is semantically equivalent to a Fortran procedure
pointer in LLVM IR level. That is, a Fortran procedure will be converted to a
opaque pointer in LLVM IR level, which is the same for a C function pointer;
a Fortran procedure pointer will be converted to a opaque pointer pointing to
a opaque pointer, which is the same for a pointer to a C function pointer.

The tests should include the following
- function result, subroutine/function arguments with varying types
  - non-character scalar
  - character (assumed-length and non-assumed-length)
  - array (static and dynamic)
  - character array
  - derived type
  - ... (polymorphic?)
- internal/external/module procedure or a C function as the target
  - procedure pointer initialization
  - procedure pointer assignment
- procedure pointer, procedure pointer target passed to a C function
- procedure pointer, procedure pointer target passed to a Fortran procedure
- procedure pointer component in derived types

---

## Current TODOs
Current list of TODOs in lowering:
- `flang/lib/Lower/CallInterface.cpp:708`: not yet implemented: procedure pointer result not yet handled
- `flang/lib/Lower/CallInterface.cpp:961`: not yet implemented: procedure pointer arguments
- `flang/lib/Lower/CallInterface.cpp:993`: not yet implemented: procedure pointer results
- `flang/lib/Lower/ConvertExpr.cpp:1119`: not yet implemented: procedure pointer component in derived type assignment
- `flang/lib/Lower/ConvertType.cpp:228`: not yet implemented: procedure pointers
- `flang/lib/Lower/Bridge.cpp:2438`: not yet implemented: procedure pointer assignment
- `flang/lib/Lower/ConvertVariable.cpp:348`: not yet implemented: procedure pointer component default initialization
- `flang/lib/Lower/ConvertVariable.cpp:416`: not yet implemented: procedure pointer globals
- `flang/lib/Lower/ConvertVariable.cpp:1459`: not yet implemented: procedure pointers
- `flang/lib/Lower/HostAssociations.cpp:162`: not yet implemented: capture procedure pointer in internal procedure
- lowering of procedure pointers in ASSOCIATED, NULL, and C_F_PROCPOINTER

Current list of TODOs in code generation:

NOTE: There are any number of possible implementations.

BoxedProcedure pass

or

- `flang/lib/Optimizer/CodeGen/TypeConverter.h:64` TODO: BoxProcType type conversion
- `flang/lib/Optimizer/CodeGen/CodeGen.cpp:2080` not yet implemented: fir.emboxproc codegen
- `flang/lib/Optimizer/CodeGen/CodeGen.cpp:629` not yet implemented: fir.boxproc_host codegen
- `flang/lib/Optimizer/CodeGen/CodeGen.cpp:1078` not yet implemented: fir.len_param_index codegen
- `flang/lib/Optimizer/CodeGen/CodeGen.cpp:3166` not yet implemented: fir.unboxproc codegen

---

Resources:
- [1] Fortran standard

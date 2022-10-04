# Polymorphic Entities

A polymorphic entity is a data entity that can be of different type during the
execution of a program.

This document aims to give insights at the representation of polymorphic
entities in FIR and how polymorphic related constructs and features are lowered
to FIR.

## Fortran standard

Here is a list of the sections and constraints of the Fortran standard involved
for polymorphic entities.

- 7.3.2.1 - 7.3.2.2: TYPE specifier (TYPE(*))
  - C708
  - C709
  - C710
  - C711
- 7.3.2.3: CLASS specifier
- 7.5.4.5: The passed-object dummy argument
  - C760
- 9.7.1: ALLOCATE statement
  - C933
- 9.7.2: NULLIFY statement
   - When a NULLIFY statement is applied to a polymorphic pointer (7.3.2.3),
     its dynamic type becomes the same as its declared type.
- 10.2.2.3: Data pointer assignment
- 11.1.3: ASSOCIATE construct
- 11.1.11: SELECT TYPE construct
  - C1157
  - C1158
  - C1159
  - C1160
  - C1161
  - C1162
  - C1163
  - C1164
  - C1165
- 16.9.76 EXTENDS_TYPE_OF (A, MOLD)
- 16.9.165 SAME_TYPE_AS (A, B)
- 16.9.184 STORAGE_SIZE (A [, KIND])
- C.10.5 Polymorphic Argument Association (15.5.2.9)

---

## Representation in FIR

### Polymorphic entities `CLASS(type1)`

A polymorphic entity is represented as a class type in FIR. In the example below
the dummy argument `p` is passed to the subroutine `foo` as a polymorphic entity
with the extensible type `point`. The type information captured in the class is
the best statically available at compile time.
`!fir.class` is a new type introduced for polymorphic entities. It's similar to
a box type but allows the distinction between a monomorphic and a polymorphic
descriptor.
A specific `BoxTypeInterface` (TypeInterface) can be introduced to share the
same API for both types where it is necessary. `!fir.class` and `!fir.box` can
also be based on a same `BaseBoxType` similar to the `BaseMemRefType` done for
MemRef.

**Fortran**
```fortran
type point
  real :: x, y
end type point

type, extends(point) :: point_3d
  real :: z
end type

subroutine foo(p)
  class(point) :: p
  ! code of the subroutine
end subroutine
```

**FIR**
```c
func.func @foo(%p : !fir.class<!fir.type<_QTpoint{x:f32,y:f32}>>)
```

### Unlimited polymorphic entities `CLASS(*)`

The unlimited polymorphic entity is represented as a class type with `none` as
element type.

**Fortran**
```fortran
subroutine bar(x)
  class(*) :: x
  ! code of the subroutine
end subroutine
```

**FIR**
```c
func.func @bar(%x : !fir.class<none>)
```

### Assumed-type `TYPE(*)`

Assumed type is added in Fortran 2018 and it is available only for dummy
arguments. It's mainly used for interfaces to non-Fortran code and is similar
to C's `void`.
An entity that is declared using the `TYPE(*)` type specifier is assumed-type
and is an unlimited polymorphic entity. It is not declared to have a type, and
is not considered to have the same declared type as any other entity,
including another unlimited polymorphic entity. Its dynamic type and type
parameters are assumed from its effective argument (7.3.2.2 - 3).

Assumed-type is represented in FIR as `!fir.box<none>`.

### SELECT TYPE construct

The `SELECT TYPE` construct select for execution at most one of its constituent
block. The selection is based on the dynamic type of the selector.

**Fortran**
```fortran
type point
  real :: x, y
end type point
type, extends(point) :: point_3d
  real :: z
end type point_3d
type, extends(point) :: color_point
  integer :: color
end type color_point

type(point), target :: p
type(point_3d), target :: p3
type(color_point), target :: c
class(point), pointer :: p_or_c
p_or_c => c
select type ( a => p_or_c )
class is (point)
  print*, a%x, a%y
type is (point_3d)
  print*, a%x, a%y, a%z
class default
  print*,
end select
```

From the Fortran standard:
> A `TYPE IS` type guard statement matches the selector if the dynamic type
and kind type parameter values of the selector are the same as those specified
by the statement. A `CLASS IS` type guard statement matches the selector if the
dynamic type of the selector is an extension of the type specified by the
statement and the kind type parameter values specified by the statement are the
same as the corresponding type parameter values of the dynamic type of the
selector.

In the example above the `CLASS IS` type guard is matched.

The construct is lowered to a specific FIR operation `fir.select_type`. It is
similar to other FIR "select" operations such as `fir.select` and
`fir.select_rank`. The dynamic type of the selector value is matched against a
list of type descriptor. The `TYPE IS` type guard statement is represented by a
`#fir.type_is` attribute and the `CLASS IS` type guard statement is represented
by a `#fir.class_is` attribute.
The `CLASS DEFAULT` type guard statement is represented by a `unit` attribute.

**FIR**
```
fir.select_type %p : !fir.class<!fir.type<_QTpoint{x:f32,y:f32}>> [
  #fir.class_is<!fir.type<_QTpoint{x:f32,y:f32}>>, ^bb1,
  #fir.type_is<!fir.type<_QTpoint_3d{x:f32,y:f32,z:f32}>>, ^bb2,
  unit, ^bb3]
```

Lowering of the `fir.select_type` operation will produce a if-then-else ladder.
The testing of the dynamic type of the selector is done by calling runtime
functions.

The runtime has two functions to compare dynamic types . Note that this two
functions _ignore_ the values of `KIND` type parameters. A version of these
functions that does not _ignore_ the value of the `KIND` type parameters will
be implemented for the  `SELECT TYPE` type guards testing.

Currently available functions for the `EXTENDS_TYPE_OF` and `SAME_TYPE_AS`
intrinsics (`flang/include/flang/Evaluate/type.h`).
```cpp
std::optional<bool> ExtendsTypeOf(const DynamicType &) const;
std::optional<bool> SameTypeAs(const DynamicType &) const;
```

**FIR** (lower level FIR/MLIR after conversion to an if-then-else ladder)
```
module  {
  func @f(%arg0: !fir.class<*>) -> i32 {
    %c4_i32 = arith.constant 4 : i32
    %c8_i32 = arith.constant 8 : i32
    %c16_i32 = arith.constant 16 : i32
    %0 = fir.gentypedesc !fir.tdesc<!fir.type<!fir.type<_QTpoint{x:f32,y:f32}>>>
    %1 = fir.convert %arg0 : (!fir.class<!fir.type<_QTpoint{x:f32,y:f32}>>) -> !fir.box<none>
    %2 = fir.convert %0 : (!fir.tdesc<!fir.type<!fir.type<_QTpoint{x:f32,y:f32}>>>) -> !fir.ref<none>
    %3 = fir.call @ExtendsTypeOfWithKind(%1, %2) : (!fir.box<none>, !fir.ref<none>) -> i1
    cond_br %3, ^bb2(%c4_i32 : i32), ^bb1
  ^bb1:  // pred: ^bb0
    %4 = fir.gentypedesc !fir.type<_QTpoint_3d{x:f32,y:f32,z:f32}>
    %5 = fir.convert %arg0 : (!fir.class<!fir.type<_QTpoint{x:f32,y:f32}>>) -> !fir.box<none>
    %6 = fir.convert %4 : (!fir.tdesc<!fir.type<_QTpoint_3d{x:f32,y:f32,z:f32}>>) -> !fir.ref<none>
    %7 = fir.call @SameTypeAsWithKind(%5, %6) : (!fir.box<none>, !fir.ref<none>) -> i1
    cond_br %7, ^bb4(%c16_i32 : i32), ^bb3
  ^bb2(%8: i32):  // pred: ^bb0
    return %8 : i32
  ^bb3:  // pred: ^bb1
    br ^bb5(%c8_i32 : i32)
  ^bb4(%9: i32):  // pred: ^bb1
    %10 = arith.addi %9, %9 : i32
    return %10 : i32
  ^bb5(%11: i32):  // pred: ^bb3
    %12 = arith.muli %11, %11 : i32
    return %12 : i32
  }
  func private @ExactSameTypeAsWithKind(!fir.box<none>, !fir.ref<none>) -> i1
  func private @SameTypeAsWithKind(!fir.box<none>, !fir.ref<none>) -> i1
}
```

Note: some dynamic type checks can be inlined for performance. Type check with
intrinsic types when dealing with unlimited polymorphic entities is an ideal
candidate for inlined checks.

---

## Dynamic dispatch

Dynamic dispatch is the process of selecting which implementation of a
polymorphic procedure to call at runtime. The runtime already has information
to be used in this process (more information can be found here:
[RuntimeTypeInfo.md](RuntimeTypeInfo.md)).

The declaration of the data structures are present in
`flang/runtime/type-info.h`.

In the example below, there is a basic type `shape` with two type extensions
`triangle` and `rectangle`.
The two type extensions override the `get_area` type-bound procedure.

**UML**
```

                          |---------------------|
                          |        Shape        |
                          |---------------------|
                          | + color:integer     |
                          | + isFilled:logical  |
                          |---------------------|
                          | + init()            |
                          | + get_area():real   |
                          |---------------------|
                                     /\
                                    /__\
                                     |
            |---------------------------------------------------|
            |                                                   |
            |                                                   |
|---------------------|                              |---------------------|
|      triangle       |                              |      rectangle      |
|---------------------|                              |---------------------|
| + base:real         |                              | + length:real       |
| + height:real       |                              | + width:real        |
|---------------------|                              |---------------------|
| + get_area():real   |                              | + get_area():real   |
|---------------------|                              |---------------------|

```

**Fortran**
```fortran
module geometry
type :: shape
  integer :: color
  logical :: isFilled
contains
  procedure :: get_area => get_area_shape
  procedure :: init => init_shape
end type shape

type, extends(shape) :: triangle
  real :: base
  real :: height
contains
  procedure :: get_area => get_area_triangle
end type triangle

type, extends(shape) :: rectangle
  real :: length
  real :: width
contains
  procedure :: get_area => get_area_rectangle
end type rectangle

type shape_array
  class(shape), allocatable :: item
end type

contains

function get_area_shape(this)
  real :: get_area_shape
  class(shape) :: this
  get_area_shape = 0.0
end function

subroutine init_shape(this, color)
  class(shape) :: this
  integer :: color
  this%color = color
  this%isFilled = .false.
end subroutine

function get_area_triangle(this)
  real :: get_area_triangle
  class(triangle) :: this
  get_area_triangle = (this%base * this%height) / 2
end function

function get_area_rectangle(this)
  real :: get_area_rectangle
  class(rectangle) :: this
  get_area_rectangle = this%length * this%width
end function

function get_all_area(shapes)
  real :: get_all_area
  type(shape_array) :: shapes(:)
  real :: sum
  integer :: i

  get_all_area = 0.0

  do i = 1, size(shapes)
    get_all_area = get_all_area + shapes(i)%item%get_area()
  end do
end function

subroutine set_base_values(sh, v1, v2)
  class(shape) :: sh
  real, intent(in) :: v1, v2

  select type (sh)
  type is (triangle)
    sh%base = v1
    sh%height = v2
  type is (rectangle)
    sh%length = v1
    sh%width = v2
  class default
    print*,'Cannot set values'
  end select
end subroutine

end module

program foo
  use geometry

  real :: area

  type(shape_array), dimension(2) :: shapes

  allocate (triangle::shapes(1)%item)
  allocate (rectangle::shapes(2)%item)

  do i = 1, size(shapes)
    call shapes(i)%item%init(i)
  end do

  call set_base_values(shapes(1)%item, 2.0, 1.5)
  call set_base_values(shapes(2)%item, 5.0, 4.5)

  area = get_all_area(shapes)

  print*, area

  deallocate(shapes(1)%item)
  deallocate(shapes(2)%item)
end program
```

The `fir.dispatch` operation is used to perform a dynamic dispatch. This
operation is comparable to the `fir.call` operation but for polymorphic
entities.
Call to `NON_OVERRIDABLE` type-bound procedure are resolved at compile time and
a `fir.call` operation is emitted instead of a `fir.dispatch`.
When the type of a polymorphic entity can be fully determined at compile
time, a `fir.dispatch` op can even be converted to a `fir.call` op. This will
be discussed in more detailed later in the document in the devirtualization
section.

**FIR**
Here is simple example of the `fir.dispatch` operation. The operation specify
the binding name of the type-bound procedure to be called and pass the
descriptor as argument. If the `NOPASS` attribute is set then the descriptor is
not passed as argument when lowered. If `PASS(arg-name)` is specified, the
`fir.pass` attribute is added to point to the PASS argument in the
`fir.dispatch` operation. `fir.nopass` attribute is added for the `NOPASS`. The
descriptor still need to be present in the `fir.dispatch` operation for the
dynamic dispatch. The CodeGen will then omit the descriptor in the argument
of the generated call.

The dispatch explanation focus only on the call to `get_area()` as seen in the
example.

**Fortran**
```fortran
get_all_area = get_all_area + shapes(i)%item%get_area()
```

**FIR**
```c
%1 = fir.convert %0 : (!fir.ref<!fir.class<!fir.type<_QMgeometryTtriangle{color:i32,isFilled:!fir.logical<4>,base:f32,height:f32>>>) -> !fir.ref<!fir.box<none>>
%2 = fir.dispatch "get_area"(%1) : (!fir.ref<!fir.box<none>>) -> f32
```

The type information is stored in the `f18Addendum` of the descriptor. The
format is defined in `flang/runtime/type-info.h` and part of its representation
in LLVM IR is shown below. The binding is comparable to a vtable. Each derived
type has a complete type-bound procedure table in which all of the bindings of
its ancestor types appear first.

**LLVMIR**

Representation of the derived type information with the bindings.
```c
%_QM__fortran_type_infoTderivedtype = type { { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, { ptr, i64, i32, i8, i8, i8, i8 }, i64, { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, i32, i8, i8, i8, i8, [4 x i8] }
%_QM__fortran_type_infoTbinding = type { %_QM__fortran_builtinsT__builtin_c_funptr, { ptr, i64, i32, i8, i8, i8, i8 } }
%_QM__fortran_builtinsT__builtin_c_funptr = type { i64 }
```

The `fir.dispatch` is then lowered to use the runtime information to extract the
correct function from the vtable and to perform the actual call. Here is
what it can look like in pseudo LLVM IR code.

**LLVMIR**
```c
// Retrieve the bindings (vtable) from the type information from the descriptor
%1 = call %_QM__fortran_type_infoTbinding* @_FortranAGetBindings(%desc)
// Retrieve the position of the specific bindings in the table
%2 = call i32 @_FortranAGetBindingOffset(%1, "get_area")
// Get the binding from the table
%3 = getelementptr %_QM__fortran_type_infoTbinding, %_QM__fortran_type_infoTbinding* %1, i32 0, i32 %2
// Get the function pointer from the binding
%4 = getelementptr %_QM__fortran_builtinsT__builtin_c_funptr, %_QM__fortran_type_infoTbinding %3, i32 0, i32 0
// Cast func pointer
%5 = inttoptr i64 %4 to <procedure pointer>
// Load the function
%6 = load f32(%_QMgeometryTshape*)*, %5
// Perform the actual function call
%7 = call f32 %6(%_QMgeometryTshape* %shape)
```

_Note:_ functions `@_FortranAGetBindings` and `@_FortranAGetBindingOffset` are
not available in the runtime and will need to be implemented.

- `@_FortranAGetBindings` retrieves the bindings from the descriptor. The
  descriptor holds the type information that holds the bindings.
- `@_FortranAGetBindingOffset` retrieves the procedure offset in the bindings
  based on the binding name provided.

Retrieving the binding table and the offset are done separately so multiple
dynamic dispatch on the same polymorphic entities can be optimized (the binding
table is retrieved only once for multiple call).

### Passing polymorphic entities as argument

**Fortran**
```fortran
TYPE t1
END TYPE
TYPE, EXTENDS(t1) :: t2
END TYPE
```

1) Dummy argument is fixed type and actual argument is fixed type.
    - `TYPE(t1)` to `TYPE(t1)`: Nothing special to take into consideration.
2) Dummy argument is polymorphic and actual argument is fixed type. In these
   cases, the actual argument need to be boxed to be passed to the
   subroutine/function since those are expecting a descriptor.
   ```c
   func.func @_QMmod1Ps(%arg0: !fir.class<!fir.type<_QMmod1Tshape{x:i32,y:i32}>>)
   func.func @_QQmain() {
     %0 = fir.alloca !fir.type<_QMmod1Tshape{x:i32,y:i32}> {uniq_name = "_QFEsh"}
     %1 = fir.embox %0 : (!fir.ref<!fir.type<_QMmod1Tshape{x:i32,y:i32}>>) -> !fir.class<!fir.type<_QMmod1Tshape{x:i32,y:i32}>>
     fir.call @_QMmod1Ps(%1) : (!fir.class<!fir.type<_QMmod1Tshape{x:i32,y:i32}>>) -> ()
     return
   }
   ```
    - `TYPE(t1)` to `CLASS(t1)`
    - `TYPE(t2)` to `CLASS(t1)`
    - `TYPE(t1)` to `CLASS(t2)` - Invalid
    - `TYPE(t2)` to `CLASS(t2)`
3) Actual argument is polymorphic and dummy argument is fixed type. These case
   are restricted to the declared type of the polymorphic entities.
    - The simple case is when the actual argument is a scalar
      polymorphic entity passed to a non-PDT. The caller just extract the
      base address from the descriptor and pass it to the function.
    - In other cases, the caller needs to perform a copyin/copyout since it
      cannot just extract the base address of the `CLASS(T)` because it is
      likely not contiguous.
    - `CLASS(t1)` to `TYPE(t1)`
    - `CLASS(t2)` to `TYPE(t1)` - Invalid
    - `CLASS(t1)` to `TYPE(t2)` - Invalid
    - `CLASS(t2)` to `TYPE(t2)`
4) Both actual and dummy arguments are polymorphic. These particular cases are
   straight forward. The function expect polymorphic entities already.
   The boxed type is passed without change.
    - `CLASS(t1)` to `CLASS(t1)`
    - `CLASS(t2)` to `CLASS(t1)`
    - `CLASS(t1)` to `CLASS(t2)` - Invalid
    - `CLASS(t2)` to `CLASS(t2)`

### User-Defined Derived Type Input/Output

User-Defined Derived Type Input/Output allows to define how a derived-type
is read or written from/to a file.

There are 4 basic subroutines that can be defined:
- Formatted READ
- Formatted WRITE
- Unformatted READ
- Unformatted WRITE

Here are their respective interfaces:

**Fortran**
```fortran
subroutine read_formatted(dtv, unit, iotype, v_list, iostat, iomsg)
subroutine write_formatted(dtv, unit, iotype, v_list, iostat, iomsg)
subroutine read_unformatted(dtv, unit, iotype, v_list, iostat, iomsg)
subroutine write_unformatted(dtv, unit, iotype, v_list, iostat, iomsg)
```

When defined on a derived-type, these specific type-bound procedures are stored
as special bindings in the type descriptor (see `SpecialBinding` in
`flang/runtime/type-info.h`).

With a derived-type the function call to `@_FortranAioOutputDescriptor` from IO
runtime will be emitted in lowering.

**Fortran**
```fortran
type(t) :: x
write(10), x
```

**FIR**
```c
%5 = fir.call @_FortranAioBeginUnformattedOutput(%c10_i32, %4, %c56_i32) : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
%6 = fir.embox %2 : (!fir.ref<!fir.type<_QTt>>) -> !fir.class<!fir.type<_QTt>>
%7 = fir.convert %6 : (!fir.class<!fir.type<_QTt>>) -> !fir.box<none>
%8 = fir.call @_FortranAioOutputDescriptor(%5, %7) : (!fir.ref<i8>, !fir.box<none>) -> i1
%9 = fir.call @_FortranAioEndIoStatement(%5) : (!fir.ref<i8>) -> i32
```

When dealing with polymorphic entities the call to IO runtime can stay
unchanged. The runtime function `OutputDescriptor` can make the dynamic dispatch
to the correct binding stored in the descriptor.

### Finalization

The `FINAL` specifies a final subroutine that might  be executed when a data
entity of that type is finalized. Section 7.5.6.3 defines when finalization
occurs.

Final subroutines like User-Defined Derived Type Input/Output are stored as
special bindings in the type descriptor. The runtime is able to handle the
finalization with a call the the `@_FortranADestroy` function
(`flang/include/flang/Runtime/derived-api.h`).

**FIR**
```c
%5 = fir.call @_FortranADestroy(%desc) : (!fir.box<none>) -> none
```

The `@_FortranADestroy` function will take care to call the final subroutines
and the ones from the parent type.

Appropriate call to finalization have to be lowered at the right places (7.5.6.3
When finalization occurs).

### Devirtualization

Sometimes there is enough information at compile-time to avoid going through
a dynamic dispatch for a type-bound procedure call on a polymorphic entity. To
be able to perform this optimization directly in FIR the dispatch table is also
present statically with the `fir.dispatch_table` and `fir.dt_entry` operations.

Here is an example of these operations representing the dispatch tables for the
same example than for the dynamic dispatch.

**FIR**
```
fir.dispatch_table @_QMgeometryE.dt.shape {
  fir.dt_entry init, @_QMgeometryPinit_shape
  fir.dt_entry get_area, @_QMgeometryPget_area_shape
}

fir.dispatch_table @_QMgeometryE.dt.rectangle {
  fir.dt_entry init, @_QMgeometryPinit_shape
  fir.dt_entry get_area, @_QMgeometryPget_area_rectangle
}

fir.dispatch_table @_QMgeometryE.dt.triangle {
  fir.dt_entry init, @_QMgeometryPinit_shape
  fir.dt_entry get_area, @_QMgeometryPget_area_triangle
}
```

With this information, an optimization pass can replace `fir.dispatch`
operations with `fir.call` operations to the correct functions when the type is
know at compile time.

This is the case in a `type is` type-guard block as illustrated below.

**Fortran**
```fortran
subroutine get_only_triangle_area(sh)
  class(shape) :: sh
  real :: area

  select type (sh)
  type is (triangle)
    area = sh%get_area()
  class default
    area = 0.0
  end select

end subroutine
```

**FIR**

The call to `get_area` in the `type is (triangle)` guard can be replaced.
```c
%3 = fir.dispatch "get_area"(%desc)
// Replaced by
%3 = fir.call @get_area_triangle(%desc)
```

Another example would be the one below. In this case as well, a dynamic dispatch
is not necessary and a `fir.call` can be emitted instead.

**Fortran**
```fortran
real :: area
class(shape), pointer :: sh
type(triangle), target :: tr

sh => tr

area = sh%get_area()
```

Note that the frontend is already replacing some of the dynamic dispatch calls
with the correct static ones. The optimization pass is useful for cases not
handled by the frontend and especially cases showing up after some other
optimizations are applied.

### `ALLOCATE`/`DEALLOCATE` statements

The allocation and deallocation of polymorphic entities are delegated to the
runtime.
The corresponding function signatures can be found in
`flang/include/flang/Runtime/allocatable.h` and in
`flang/include/flang/Runtime/pointer.h` for pointer allocation.

`ALLOCATE`

The `ALLOCATE` statement is lowered to runtime calls as shown in the example
below.

**Fortran**
```fortran
allocate(triangle::shapes(1)%item)
allocate(rectangle::shapes(2)%item)
```

**FIR**
```c
%0 = fir.alloca !fir.class<!fir.type<_QMgeometryTtriangle{color:i32,isFilled:!fir.logical<4>,base:f32,height:f32>>
%1 = fir.alloca !fir.class<!fir.type<_QMgeometryTtriangle{color:i32,isFilled:!fir.logical<4>,base:f32,height:f32}>>
%3 = fir.convert %0 : (!fir.ref<!fir.class<!fir.type<_QMgeometryTtriangle{color:i32,isFilled:!fir.logical<4>,base:f32,height:f32>>>) -> !fir.ref<!fir.box<none>>
%4 = fir.gentypedesc !fir.type<_QMgeometryTtriangle{color:i32,isFilled:!fir.logical<4>,base:f32,height:f32}>>
%5 = fir.call @_FortranAAllocatableInitDerived(%3, %4)

%6 = fir.convert %1 : (!fir.ref<!fir.class<_QMgeometryTtriangle{color:i32,isFilled:!fir.logical<4>,base:f32,height:f32}>>>) -> !fir.ref<!fir.box<none>>
%7 = fir.gentypedesc !fir.type<_QMgeometryTtriangle{color:i32,isFilled:!fir.logical<4>,base:f32,height:f32}>> %8 = fir.call @_FortranAAllocatableInitDerived(%6, %7)
```

For pointer allocation, the `PointerAllocate` function is used.

`DEALLOCATE`

The `DEALLOCATE` statement is lowered to a runtime call to
`AllocatableDeallocate` and `PointerDeallocate` for pointers.

**Fortran**
```fortran
deallocate(shapes(1)%item)
deallocate(shapes(2)%item)
```

**FIR**
```c
%8 = fir.call @_FortranAAllocatableDeallocate(%desc1)
%9 = fir.call @_FortranAAllocatableDeallocate(%desc2)
```

### `EXTENDS_TYPE_OF`/`SAME_TYPE_AS` intrinsics

`EXTENDS_TYPE_OF` and `SAME_TYPE_AS` intrinsics have implementation in the
runtime. Respectively `SameTypeAs` and `ExtendsTypeOf` in
`flang/include/flang/Evaluate/type.h`.

Both intrinsic functions are lowered to their respective runtime calls.

### Assignment / Pointer assignment

Intrinsic assignment of an object to another is already implemented in the
runtime. The function `@_FortranAAsssign` performs the correct operations.

Available in `flang/include/flang/Runtime/assign.h`.

### User defined assignment and operator

**Fortran**
```fortran
module mod1
type t1
contains
  procedure :: assign_t1
  generic :: assignment(=) => assign_t1
end type t1

type, extends(t1) :: t2
end type

contains

subroutine assign_t1(to, from)
  class(t1), intent(inout) :: to
  class(t1), intent(in) :: from
  ! Custom code for the assignment
end subroutine

subroutine assign_t2(to, from)
  class(t2), intent(inout) :: to
  class(t2), intent(in) :: from
  ! Custom code for the assignment
end subroutine

end module

program main
use mod

class(t1), allocatable :: v1
class(t1), allocatable :: v2

allocate(t2::v1)
allocate(t2::v2)

v2 = v1

end program
```

In the example above the assignment `v2 = v1` is done by a call to `assign_t1`.
This is resolved at compile time since `t2` could not have a generic type-bound
procedure for assignment with an interface that is not distinguishable. This
is the same for user defined operators.

### `NULLIFY`

When a `NULLIFY` statement is applied to a polymorphic pointer (7.3.2.3), its
dynamic type becomes the same as its declared type.

The `NULLIFY` statement is lowered to a call to the corresponding runtime
function `PointerNullifyDerived` in `flang/include/flang/Runtime/pointer.h`.

### Impact on existing FIR operations dealing with descriptors

Currently, FIR has a couple of operations taking descriptors as inputs or
producing descriptors as outputs. These operations might need to deal with the
dynamic type of polymorphic entities.

- `fir.load`/`fir.store`
  - Currently a `fir.load` of a `fir.box` is a special case. In the code
    generation no copy is made. This could be problematic with polymorphic
    entities. When a `fir.load` is performed on a `fir.class` type, the dynamic
    can be copied.

  **Fortran**
  ```fortran
  module mod1
    class(shape), pointer :: a
  contains
  subroutine sub1(a, b)
    class(shape) :: b
    associate (b => a)
      ! Some more code
    end associate
  end subroutine
  end module
  ```

  In the example above, the dynamic type of `a` and `b` might be different. The
  dynamic type of `a` must be copied when it is associated on `b`.

  **FIR**
  ```c
  // fir.load must copy the dynamic type from the pointer `a`
  %0 = fir.address_of(@_QMmod1Ea) : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMmod1Tshape{x:i32,y:i32}>>>>
  %1 = fir.load %0 : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMmod1Tshape{x:i32,y:i32}>>>>
  ```

- `fir.embox`
  - The embox operation is used to create a descriptor from a reference. With
    polymorphic entities, it is used to create a polymorphic descriptor from
    a derived type. The declared type of the descriptor and the derived type
    are identical. The dynamic type of the descriptor must be set when it is
    created. This is already handled by lowering.

- `fir.rebox`
  - The rebox operation is used to create a new descriptor from a another
    descriptor with new optional dimension. If the original descriptor is a
    polymorphic entities its dynamic type must be propagated to the new
    descriptor.
  ```
  %0 = fir.slice %c10, %c33, %c2 : (index, index, index) -> !fir.slice<1>
  %1 = fir.shift %c0 : (index) -> !fir.shift<1>
  %2 = fir.rebox %x(%1)[%0] : (!fir.class<!fir.array<?x!fir.type<>>>, !fir.shift<1>, !fir.slice<1>) -> !fir.class<!fir.array<?x!fir.type<>>>
  ```
---

# Testing

- Lowering part is tested with LIT tests in tree
- Polymorphic entities involved a lot of runtime information so executable
  tests will be useful for full testing.

---

# Current TODOs
Current list of TODOs in lowering:
- `flang/lib/Lower/Allocatable.cpp:465` not yet implemented: SOURCE allocation
- `flang/lib/Lower/Allocatable.cpp:468` not yet implemented: MOLD allocation
- `flang/lib/Lower/Allocatable.cpp:471` not yet implemented: polymorphic entity allocation
- `flang/lib/Lower/Bridge.cpp:448` not yet implemented: create polymorphic host associated copy
- `flang/lib/Lower/Bridge.cpp:2185` not yet implemented: assignment to polymorphic allocatable
- `flang/lib/Lower/Bridge.cpp:2288` not yet implemented: pointer assignment involving polymorphic entity
- `flang/lib/Lower/Bridge.cpp:2316` not yet implemented: pointer assignment involving polymorphic entity
- `flang/lib/Lower/CallInterface.cpp:795` not yet implemented: support for polymorphic types
- `flang/lib/Lower/ConvertType.cpp:237` not yet implemented: support for polymorphic types

Current list of TODOs in code generation:

- `flang/lib/Optimizer/CodeGen/CodeGen.cpp:897` not yet implemented: fir.dispatch codegen
- `flang/lib/Optimizer/CodeGen/CodeGen.cpp:911` not yet implemented: fir.dispatch_table codegen
- `flang/lib/Optimizer/CodeGen/CodeGen.cpp:924` not yet implemented: fir.dt_entry codegen
- `flang/lib/Optimizer/CodeGen/CodeGen.cpp:2651` not yet implemented: fir.gentypedesc codegen

---

Resources:
- [1] https://www.pgroup.com/blogs/posts/f03-oop-part1.htm
- [2] https://www.pgroup.com/blogs/posts/f03-oop-part2.htm
- [3] https://www.pgroup.com/blogs/posts/f03-oop-part3.htm
- [4] https://www.pgroup.com/blogs/posts/f03-oop-part4.htm
- [5] Modern Fortran explained

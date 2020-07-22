## Burnside: The Bridge from the Fortran front-end to FIR

This document sketches the translation of various Fortran snippets from
their syntactic level to how they ought to be represented in FIR.  These
translations are representative and written in pseudo-code.

This document shows examples of how Fortran fragments might be lowered into
FIR fragments.  The style used throughout the document is to first show the
Fortran code fragment above a line and the FIR code fragment below the
line.

### Program Units (PROGRAM, MODULE, and SUBMODULE)

FIR has one flat global namespace.  The global namespace can be populated
by Ops that represent code (functions), data (variables, constants), and
auxiliary structures (dispatch tables).

Name collisions and scoping will be handled by a name mangling scheme. This
scheme ought to be a bijection from the tree of Fortran syntactic symbols
to and from the set of mangled names.

A `PROGRAM` will necessarily have its executable definition wrapped in a
FIR `func` like a `SUBROUTINE`.  Again, it is assumed the name mangling
scheme will provide a mapping to a distinct name.

### Procedures (FUNCTION and SUBROUTINE)

```fortran
    FUNCTION foo (arg1, arg2) RESULT retval

    SUBROUTINE bar (arg1, arg2)
```
----
```mlir
    func @foo(!fir.ref<!T1>, !fir.ref<!T2>) -> !TR
    func @bar(!fir.ref<!T1>, !fir.ref<!T2>)
```

MLIR is strongly typed, so the types of the arguments and return value(s),
if any, must be explicitly specified.  (Here, `arg1`, `arg2`, and `retval`
have the types `!T1`, `!T2`, and `!TR`, resp.)  Also reflected is the
default calling convention: Fortran passes arguments by reference.

#### Internal subprograms

These will be lowered as any other `SUBROUTINE`. The difference will be
that they may take an extra `tuple` reference argument to refer to
variables in the host context.  Host associated variables must be bundled
and passed explicitly on the FIR side. An example will be detailed below.

#### Statement functions

These are very simple internal subroutines, in a sense. They will be
lowered in the same way.

### Non-executable statements

#### Data

Some non-executable statements may create constant (`PARAMETER`) or
variable data.  This information should be lowered.

##### Constants

```fortran
    INTEGER, PARAMETER :: x = 1
    CHARACTER (LEN = 10), PARAMETER :: DIGITS = "0123456789"
```
----
```mlir
    %0 = constant 1 : i32

    fir.global @_QG_digits constant : !fir.array<10:!fir.char<1>> {
      constant '0' : !fir.char<1>
      ...
      constant '9' : !fir.char<1>
    }
```

##### Local Variable

```fortran
    CHARACTER (LEN = 1) :: digit
    INTEGER :: i
```
----
```mlir
    %len = constant 1 : i32
    %digit = fir.alloca !fir.char<1>, %len : !fir.ref<!fir.char<1>>
    %i = fir.alloca i32 : !fir.ref<i32>
```

Note that in MLIR, the `%` sigil denotes an ssa-value, the `@` sigil
denotes a global symbol, and the `!` sigil denotes a type.

##### Process lifetime variable

```fortran
    COMMON /X/ A(10),B(10)

    MODULE mymod
      INTEGER a

    SUBROUTINE subr()
      REAL, SAVE :: s
      DATA s/12.0/
```
----
```mlir
    fir.global @common_x : tuple<!fir.array<10 : f32>, !fir.array<10 : f32>> {}

    fir.global @mymod_a : i32 {}

    fir.global @subr_s : f32 {
      constant 12.0 : f32
    }
```

The empty initializer region could mean these variables are placed in the
`.bss` section.

#### Other non-executable statements

These statements will define other properties of how the Fortran gets
lowered.  For example, a variable in a `COMMON` block needs to reside in a
`fir.global`, or the structure of a derived type (user-defined record),
which would be reflected in a `!fir.type`.

#### A note on TYPEs

A FIR type is an synthesis of the Fortran concepts of type, attributes, and
type parameters.

##### Intrinsic types

For Fortran intrinsic types, there is a direct translation to a FIR type.

```fortran
    REAL(4) a
    COMPLEX(8) b
    CHARACTER(1,LEN=4) c
    LOGICAL(1) d
    INTEGER(4) e

    CHARACTER(1,LEN=*) f
```
----
```mlir
    %a = ... : !fir.real<4>
    %b = ... : !fir.complex<8>
    %c = ... : !fir.array<4:!fir.char<1>>
    %d = ... : !fir.logical<1>
    %e = ... : !fir.int<4>

    %f_data = ... : !fir.ref<!fir.array<?:!fir.char<1>>>
    %f_len = ... : i32
    %f = fir.emboxchar %f_data, %f_len : !fir.boxchar<1>
```

The bridge will have a mapping of what the front-end kind value must map to
in the internal representation.  For example, the f18 front-end maps kind
values for integers to the size in bytes of the integer representation.
Such mappings must be provided for all intrinsic type kind values.

The Fortran `CHARACTER` variable, `f`, is a bit more complicated as there
is both a reference to a buffer (that contains the characters) and an
extra, assumed length, `LEN` type parameter to keep track of the length of
the buffer.  The buffer is a sequence of `!fir.char<1>` values in memory.
The pair, `(buffer, len)`, may be boxed in a `!fir.boxchar<1>` type
object.

##### Derived types

Fortran also has derived types and these are supported with a more
elaborate record syntax.

```fortran
    TYPE :: person
      CHARACTER(LEN=20) :: name
      INTEGER :: age
    END TYPE

    TYPE(person) :: george
```
----
```mlir
    %george = ... : !fir.type<person {name : !fir.array<20:!fir.char<1>>, age : i32}>
```

Fortran allows the compiler to reorder the fields in the derived type.
`SEQUENCE` can be used to disable reordering. (Name mangling can provide a
compile-time distinction, as needed.)

Fortran allows a derived type to have type parameters. There are `KIND`
type parameters and `LEN` type parameters. A `KIND` type parameter is a
compile-time known constant. As such, it is possible for the compiler
implementation to create a distinct type for each set of `KIND` type
parameters (by name mangling, for instance).

The `LEN` type parameters are runtime constant and not necessarily known at
compile-time. These values must be provided when constructing a value of
derived type in FIR, just as regular fields must be provided. (That does
not preclude an optimizer from eliminating unused `LEN` parameters.)

Because of Fortran's `LEN` type parameters, an implementation is allowed to
defer the size and layout of an entity of derived type until runtime.

Lowering may also exploit ad hoc product types created as needed. This can
be done using the standard dialect `tuple` type.

##### Arrays

An entity with type _T_ and a `DIMENSION` attribute is an array with
elements of type _T_ in Fortran.

```fortran
    INTEGER arr
    DIMENSION arr(10,20)
```
----
```mlir
    %arr = ... : !fir.array<10x20 : i32>
```

A FIR array is laid out in column-major order exactly like a Fortran array.

##### Pointer and reference types

The attribute `POINTER` can be used similarly to create a pointer entity.
The `ALLOCATABLE` attribute is another Fortran attribute that can be used
to indicate an entity's storage is to be allocated at runtime. As mentiond
previosuly, Fortran uses pass-by-reference calling semantics too.

```fortran
    INTEGER, POINTER :: ptr
    REAL, ALLOCATABLE, DIMENSION(1000) :: al

    INTERFACE
      SUBROUTINE fun(ptr, al)
        INTEGER, POINTER :: p
        REAL, ALLOCATABLE :: a
      END SUBROUTINE
    END INTERFACE
```
----
```mlir
    %ptr = ... : !fir.ptr<i32>
    %al = ... : !fir.heap<!fir.array<1000 : f32>>

    func @fun(!fir.ref<!fir.ptr<i32>>, !fir.ref<!fir.heap<f32>>)
```

Note that references to pointers and heap allocatables are
allowed. However, a pointer/heap cannot point directly to a pointer/heap.

```mlir
    %err1 = ... : !fir.ptr<!fir.ptr<i32>>   // Invalid type
    %err2 = ... : !fir.heap<!fir.ptr<i32>>  // Invalid type
```

Note that a value of function type is also considered a reference.

```mlir
    %fun = ... : (i32, f64) -> i1   // %fun is a reference to a func object
```

##### Boxed types

Boxed types are reference types. A boxed entity is implicitly located in
memory.  The only way to construct a boxed value is by providing a memory
reference type, discussed above. Any reference can be emboxed.

There are additionally, two special-purpose box types. A `!fir.boxchar`
value is a `CHARACTER` variable (in memory) including both a pointer to the
buffer and the `LEN` type parameter. `boxchar` was discussed above.

The second special case is the `!fir.boxproc` type. A Fortran internal
procedure can reference variables in its host's scope. Fortran also allows
pointers to procedures. A value of type `!fir.boxproc` then is a pair of
references, one for the procedure pointer and the other a pointer to a
tuple of host associated values.

```fortran
    SUBROUTINE host
      REAL X
      PROCEDURE(), POINTER :: procptr
      ...
      procptr => intern
      ...
      CALL procptr
    CONTAINS
      SUBROUTINE intern
        X = ...
```
----
```mlir
    func @host() {
      %x = ... : !fir.ref<f32>
      ...
      %bag_val = fir.insert_value %b, %x, %0 : ... -> tuple<!fir.ref<f32>, ...>
      %bag = ... : !fir.ref<tuple<!fir.ref<f32>, ...>>
      fir.store %bag_val to %bag : !fir.ref<tuple<!fir.ref<f32>, ...>>
      %procptr = fir.emboxproc @intern, %bag : ... -> !fir.boxproc<() -> ()>
      ...
      fir.call %procptr() : () -> ()
```

Here, the call to the boxed procedure implicitly passes the extra argument, the
reference to `%bag`, which contains the value of the variable `x`.

##### Miscellaneous types

Fortran uses triple notation to describe array sections, strided views of
multidimensional arrays.  These sections can be captured using the
`fir.gendims` instruction which produces a value of type `!fir.dims<n>`.

```fortran
    DIMENSION (10,10) a
    ... a(2:6:2,1:7:4) ...
```
----
```mlir
    // the following line is pseudocode
    %1 = fir.gendims 2,6,2, 1,7,4 : !fir.dims<2>
```

Fortran also allows the implementation to reorder fields in a derived
type. Furthermore, the sizes of these fields and the layout may be left up
to the runtime. This could mean that the backend needs to generate runtime
calls to determine the offsets and sizes of fields.

```fortran
    TYPE ding(k)
      ...
      TYPE(T(k)) :: field_name
```
----
```mlir
    %2 = fir.field("field_name") : !fir.field
```

When lowering a boxed value, the compiler may need to test what the exact
type of the value is at runtime. (For example, when generating code for
`SELECT TYPE`.)

```fortran
    CLASS(*) :: x
    SELECT TYPE (x)
      ...
```
----
```mlir
    %1 = fir.box_tdesc %x : (!fir.box<none>) -> !fir.tdesc<none>
```

The `none` type is used when the entity has unlimited polymorphic type. See
below for a larger example of `SELECT TYPE`.

### Executable statements

The main purpose of lowering is to lower all the executable statements from
Fortran into FIR in a semantics preserving way.

#### Substrings

```fortran
    ID(4:9)
```
----
```mlir
    %id = ... : !fir.ref<!fir.array<20 : !fir.char<1>>>
    %1 = fir.coordinate_of %id, %c3 : ... -> !fir.ref<!fir.char<1>>
    %2 = fir.emboxchar %1, %c5 : ... -> !fir.boxchar<1>
```

#### Structure components

```fortran
    scalar_parent%scalar_field
```
----
```mlir
    %sf = fir.field("scalar_field") : !fir.field
    %1 = fir.coordinate_of %scalar_parent, %sf : ... -> !fir.ref<f32>
```

#### Type parameters

```fortran
    TYPE ding(dim)
      INTEGER, LEN :: dim
      REAL :: values(dim)
    END TYPE ding

    ding(x) :: a_ding
    ... a_ding%dim ... 
```
----
```mlir
    %1 = fir.len_param_index("dim") : !fir.field
    %2 = fir.coordinate_of %a_ding, %1 : ... -> !fir.ref<i32>
    %3 = fir.load %2 : !fir.ref<i32>
```

#### Arrays

```fortran
    ... A ...         ! whole array
    ... B(4) ...      ! array element
    ... C(1:10) ...   ! array section
    ... D(1:10:2) ... ! array section with stride
    INTEGER, DIMENSION :: V(4)
    ... E(V) ...      ! array section with vector subscript
```
----
```mlir
    %1 = fir.load %a : !fir.ref<!fir.array<*:f32>>

    %2 = fir.extract_element %b, %c4 : (!fir.array<?:f32>, i32) -> f32

    %3 = fir.coordinate_of %c, %c1 : (!fir.ref<!fir.array<?:f32>>, i32) -> !fir.ref<f32>
    %4 = fir.convert %3 : (!fir.ref<f32>) -> !fir.ref<!fir.array<10:f32>>
    %5 = fir.load %4 : (!fir.ref<!fir.array<10:f32>>) -> !fir.array<10:f32>

    %6 = fir.gendims %c1, %c10, %c2 : (i32, i32, i32) -> !fir.dims<1>
    %7 = fir.embox %d, %6 : (!fir.ref<!fir.array<40:f32>>, !fir.dims<1>) -> !fir.embox<!fir.array<10:f32>>

    // create a temporary to hold E(V)
    %v = ... : !fir.array<4:i32>
    %8 = fir.alloca !fir.array<4:f32> : !fir.ref<!fir.array<4:f32>>
    fir.do_loop %i = %c1 to %c4 unordered {
      %9 = fir.extract_value %v, %i : (!fir.array<4:i32>, index) -> i32
      %10 = fir.extract_value %e, %9 : (!fir.array<?:f32>, i32) -> f32
      %11 = fir.coordinate_of %8, %i : (!fir.ref<!fir.array<4:f32>>, index) -> !fir.ref<f32>
      fir.store %10 to %11 : !fir.ref<f32>
    }
```

In the fourth case, lowering could also create a temporary and copy the
values from the section `D(1:10:2)` into it, but the preference should be
to defer copying data until it is necessary (as in the fifth non-affine
case, `E(V)`).

#### Image selector

```fortran
    REAL :: A(10)[5,*]

    ... A(:)[1,4] ...    ! selects image 16 (if available)
```
----
```mlir
    %1 = fir.call @runtime_fetch_array(%a, %c_1, %c_4, ...) : (!fir.box<!fir.array<10:f32>>, i32, i32, ...) -> !fir.ref<!fir.array<10:f32>>
```

#### Dynamic association

```fortran
    ALLOCATE (x(n), b(-3:m, 0:9))

    NULLIFY (p)

    DEALLOCATE (x, b)
```
----
```mlir
    %x = fir.allocmem f32, %n : !fir.heap<!fir.array<?:f32>>

    %c4 = constant 4 : i32
    %1 = addi %m, %c4 : i32
    %2 = constant 10 : i32
    %b = fir.allocmem f32, %1, %2 : !fir.heap<!fir.array<?x10:f32>>

    %zero = constant 0 : i64
    %null = fir.convert %zero : (i64) -> !fir.ptr<none>
    fir.store %null to %p : !fir.ref<!fir.ptr<none>>

    fir.freemem %x : !fir.heap<!fir.array<?:f32>>
    fir.freemem %b : !fir.heap<!fir.array<?x10:f32>>
```

#### Basic operators

Operators like `**`, `*`, `/`, etc. will be lowered into standard dialect
operations or runtime calls as needed.

```fortran
    a * b
    c .LE. d
```
----
```mlir
    %0 = mulf %a, %b : f32
    %1 = cmp "le" %c, %d : (f32, f32) -> i1
```

#### Calls

```fortran
    CALL foo(v1)
    ... func(v2, v3) ...

    pp => bar
    CALL pp(v4)

    CALL object%method(arg)
```
----
```mlir
    fir.call @foo(%v1) : (!fir.ref<!T1>) -> ()
    %1 = fir.call @func(%v2, %v3) : (!fir.ref<!T2, !T3>) -> i64

    %pp = fir.address_of(@bar) : ((!fir.ref<i64>) -> ()) -> !fir.ref<(!fir.ref<i64>) -> ()>
    fir.icall %pp(%v4) : (!fir.ref<i64>) -> ()

    fir.dispatch "method"(%object, %arg) : (!fir.box<!fir.type<mytype{f1:i64, f2:f64}>>, !fir.ref<i64>) -> ()
```

There are two modes of argument passing in Fortran: calls that are "Fortran
77" style and use an implicit interface, and calls that require an
interface. In FIR, this translates to passing a simple reference to an
entity's data versus passing a boxed reference value. The following calls
illustrate this distinction.

```fortran
    SUBROUTINE sub1(a)
      INTEGER :: a(10,10)  ! F77 style
      ...
    INTERFACE
      SUBROUTINE sub2(a)
        INTEGER :: a(:,:)    ! assumed shape
      ...
    PROGRAM p
      INTEGER :: a(10,10)
      CALL sub1(a)
      CALL sub2(a)
```
----
```mlir
    func @sub1(!fir.ref<!fir.array<10x10:i32>>) -> ()
    func @sub1(!fir.box<!fir.array<?x?:i32>>) -> ()

    func @_QP_p() {
      %c1 = constant 1 : i32
      %c10 = constant 10 : i32
      %a1 = fir.alloca !fir.array<10x10:i32> : !fir.ref<!fir.array<10x10:i32>>
      fir.call @sub1(%a1) : (!fir.ref<!fir.array<10x10:i32>>) -> ()
      %1 = fir.gendims %c1, %c10, %c1, %c1, %c10, %c1 : (i32,i32,i32,i32,i32,i32) -> !fir.dims<2>
      %a2 = fir.embox %a1, %1 : (!fir.ref<!fir.array<10x10:i32>>, !fir.dims<2>) -> !fir.box<!fir.array<?x?:i32>>
      fir.call @sub2(%a2) : (!fir.box<!fir.array<?x?:i32>>) -> ()
```

When lowering into FIR, the bridge must explicitly perform any allocation,
copying, deallocation, and finalization on temporary entities as required
by the Fortran standard, preserving the copy-in copy-out calling
convention.

#### Parentheses (10.1.8)

```fortran
    (a + b) + (a + c)  ! cannot rewrite as (2 * a) + b + c
```
----
```mlir
    %1 = addf %a, %b : f32
    %2 = fir.no_reassoc %1 : f32  // to prevent reassociation
    %3 = addf %a, %c : f32
    %4 = fir.no_reassoc %3 : f32
    %5 = addf %2, %4 : f32
```

One must also specify to LLVM that these operations will not be reassociated.

#### Assignment

```fortran
    scalar = e1   ! intrinsic scalar assignment
    array = e2    ! intrinsic array assignment
    object = e3   ! defined assignment
    pntr => e4    ! pointer assignment
    pproc => func ! procedure pointer assignment
```
----
```mlir
    %e1 = ... : f32
    fir.store %e1 to %scalar : !fir.ref<f32>
    
    %e2 = ... : !fir.array<10x10 : i32>
    fir.store %e2 to %array : !fir.ref<!fir.array<10x10 : i32>>

    %e3 = ... !fir.ref<!T>
    %object = ... !fir.ref<!U>
    fir.call @defd_assn(%object, %e3) : ... -> ()

    %e4 = ... : !fir.ptr<!V>
    %pntr = ... : !fir.ref<!fir.ptr<!V>>
    fir.store %e4 to %pntr : !fir.ref<!fir.ptr<!V>>

    @func(i32, i32) -> i32
    %fn = fir.address_of(@func) : ((i32, i32) -> i32) -> !fir.ptr<(i32, i32) -> i32>
    %pproc = ... : !fir.ref<!fir.ptr<(i32, i32) -> i32>>
    fir.store %fn to %pproc : !fir.ref<!fir.ptr<(i32, i32) -> i32>>
```

#### Masked assignment

```fortran
    WHERE (arr < threshold)
      arr = arr + increment
    ELSEWHERE
      arr = threshold
    END WHILE
```
----
```mlir
    %arr = ... : !fir.array<?:!T>
    %threshold = ... : !fir.array<?:!T>
    fir.do_loop %i = %c1 to %size {
      %arr_i = fir.extract_value %arr, %i : ... -> !T
      %threshold_i = fir.extract_value %threshold, %i : ... -> !T
      %1 = cmp "lt" %arr_i, %threshold_i : ... -> i1
      fir.where %1 {
        %2 = addf %arr_i, %increment : !T
	%3 = fir.coordinate_of %arr, %i : ... -> !fir.ref<!T>
	fir.store %2 to %3 : !fir.ref<!T>
      } otherwise {
	%4 = fir.coordinate_of %arr, %i : ... -> !fir.ref<!T>
        fir.store %threshold_i to %4
      }
    }
```

#### FORALL

```fortran
    FORALL (i = 1:100)
      a(i) = b(i) / c(i)
    END FORALL
```
----
```mlir
    fir.do_loop %i = %c1 to %c100 unordered {
      %1 = fir.extract_value %b, %i : (!fir.array<?:f32>, index) -> f32
      %2 = fir.extract_value %c, %i : (!fir.array<?:f32>, index) -> f32
      %3 = divf %1, %2 : f32
      %4 = fir.coordinate_of %a, %i : (!fir.ref<!fir.array<?:f32>>, index) -> !fir.ref<f32>
      fir.store %3 to %4 : !fir.ref<f32>
    }
```

#### ASSOCIATE construct

```fortran
    ASSOCIATE (z => EXP(-(x**2+y**2)) * COS(theta))
      CALL foo(z)
    END ASSOCIATE
```
----
```mlir
    %1 = ... : f32
    %2 = fir.call @exp(%1) : (f32) -> f32
    %3 = fir.load %theta : !fir.ref<f32>
    %4 = fir.call @cos(%3) : (f32) -> f32
    %5 = mulf %2, %4 : f32
    fir.store %5 to %z : !fir.ref<f32>
    fir.call @foo(%z) : (!fir.ref<f32>) -> ()
```

#### DO construct

```fortran
    DIMENSION a(10,10,10), b(10,10,10)

    DO i = 1, m
      DO j = 1, n
        c(i,j) = dot_prod(a(i,j,:), b(:,i,j))
      END DO
    END DO
```
----
```mlir
    %c1 = constant 1 : index
    %c10 = constant 10 : index
    %c100 = constant 100 : index
    %c1000 = constant 1000 : index
    %1 = fir.gendims %c1, %c1000, %c100 : !fir.dims<1>
    %2 = fir.gendims %c1, %c10, %c1 : !fir.dims<1>

    fir.do_loop %i = %c1 to %m {
      fir.do_loop %i = %c1 to %n {
        %13 = fir.coordinate_of %a, %i, %j : !fir.ref<!fir.array<?:f32>>
        %14 = fir.embox %13, %1 : (!fir.ref<!fir.array<?:f32>>, !fir.dims<1>) -> !fir.box<!fir.array<?:f32>>
        %15 = fir.coordinate_of %b, %c1, %i, %j : !fir.ref<f32>
        %16 = fir.convert %15 : (!fir.ref<f32>) -> !fir.ref<!fir.array<?:f32>>
        %17 = fir.embox %16, %2 : (!fir.ref<!fir.array<?:f32>>, !fir.dims<1>) -> !fir.box<!fir.array<?:f32>>
        %18 = fir.call @dot_prod(%14, %17) : (!fir.box<!fir.array<?:f32>>, !fir.box<!fir.array<?:f32>>) -> f32
        %19 = fir.coordinate_of %c, %i, %j : (!fir.box<!fir.array<?:?:f32>>, index, index) -> !fir.ref<f32>
	fir.store %18 to %19 : !fir.ref<f32>
      }
    }
```

In this lowering, the array sections from the arrays `a` and `b` are _not_
copied to a temporary memory buffer, but are instead captured in boxed
values (`%14` and `%17`).

#### IF construct

```fortran
    IF (a > 0) THEN
      ...
    ELSE
      ...
    END IF
```
----
```mlir
    %1 = ... : i1
    cond_br %1, ^bb1(%2:i32), ^bb2(%3:i32)
```

#### SELECT CASE construct

```fortran
    SELECT CASE (p)
    CASE (1, 3:5)
      ...
    CASE (:-1)
      ...
    CASE (10:)
      ...
    CASE DEFAULT
      ...
    END SELECT CASE
```
----
```mlir
    fir.select_case %p : i32 [#fir.point,%c1,^bb1, #fir.interval,%c3,%c5,^bb1, #fir.upper,%cn1,^bb2, #fir.lower,%c10,^bb3, unit,^bb4]
```

#### SELECT RANK construct

```fortran
    SELECT RANK (p)
    RANK (2)
      ...
    RANK (*)
      ...
    RANK DEFAULT
      ...
    END SELECT RANK
```
----
```mlir
    fir.select_rank %p : i32 [2,^bb1(%1:f32), -1,^bb2, unit,^bb3(%2:f32,%3:i32)]
```

#### SELECT TYPE construct

```fortran
    SELECT TYPE (p)
    TYPE IS (type_a)
      ...
    CLASS IS (super_b)
      ...
    CLASS DEFAULT
      ...
    END SELECT TYPE
```
----
```mlir
    fir.select_type %p : !fir.box<none> [#fir.instance<!fir.type<type_a ...>>,^bb_1(%1:i32,%2:i64), #fir.subsumed<!fir.type<super_b ...>>,^bb_2(%3:f32,%4:f64,%5:i32), unit,^bb_3]
```
----
```mlir
     %type_a_desc = fir.gentypedesc !fir.type<type_a ...> : !fir.tdesc<!fir.type<type_a ...>>
     %super_b_desc = fir.gentypedesc !fir.type<super_b ...> : !fir.tdesc<!fir.type<super_b ...>>
     %11 = fir.box_tdesc %p : (!fir.box<none>) -> !fir.tdesc<none>
     %12 = cmp "eq" %11, %type_a_desc : (!fir.tdesc<none>, !fir.tdesc<!fir.type<type_a ...>>) -> i1
     cond_br %2, ^bb1(%1:i32,%2:i64), ^bb1b(%3:f32,%4:f64,%5:i32)
    ^bb1(%a1,%a2 : i32,i64):
     ...
    ^bb1b(%b1,%b2,%b3 : f32,f64,i32):
     %13 = fir.call @is_subtype_of(%11, %super_b_desc) : ... -> i1
     cond_br %13, ^bb2(%b1,%b2,%b3), ^bb3 
    ^bb2(%b1,%b2,%b3 : f32,f64,i32):
     ...
    ^bb3:
     ...
```

#### Jumping statements

```fortran
    STOP
    ERROR STOP
    FAIL IMAGE
    CONTINUE loop
    EXIT a_construct
    GOTO label1
    GOTO (label2,label3,label4), i
```
----
```mlir
    fir.call @stop()
    fir.unreachable

    fir.call @error_stop()
    fir.unreachable

    fir.call @fail_image()
    fir.unreachable

    br ^bb_continue

    br ^bb_exit

    br ^bb_label1

    fir.select %i : i32 [1,^bb_label2(%1:i32), 2,^bb_label3, 3,^bb_label4, unit,^fallthru]
    ^fallthru:
```


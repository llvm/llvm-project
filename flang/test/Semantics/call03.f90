! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
! Test 15.5.2.4 constraints and restrictions for non-POINTER non-ALLOCATABLE
! dummy arguments.

module m01
  type :: t
  end type
  type :: pdt(n)
    integer, len :: n
  end type
  type :: pdtWithDefault(n)
    integer, len :: n = 3
  end type
  type :: tbp
   contains
    procedure :: binding => subr01
  end type
  type :: final
   contains
    final :: subr02
  end type
  type :: alloc
    real, allocatable :: a(:)
  end type
  type :: ultimateCoarray
    real, allocatable :: a[:]
  end type

 contains

  subroutine subr01(this)
    class(tbp), intent(in) :: this
  end subroutine
  subroutine subr02(this)
    type(final), intent(inout) :: this
  end subroutine

  subroutine poly(x)
    class(t), intent(in) :: x
  end subroutine
  subroutine polyassumedsize(x)
    class(t), intent(in) :: x(*)
  end subroutine
  subroutine assumedsize(x)
    real :: x(*)
  end subroutine
  subroutine assumedrank(x)
    real :: x(..)
  end subroutine
  subroutine assumedtypeandsize(x)
    type(*) :: x(*)
  end subroutine
  subroutine assumedshape(x)
    real :: x(:)
  end subroutine
  subroutine contiguous(x)
    real, contiguous :: x(:)
  end subroutine
  subroutine intentout(x)
    real, intent(out) :: x
  end subroutine
  subroutine intentout_arr(x)
    real, intent(out) :: x(:)
  end subroutine
  subroutine intentinout(x)
    real, intent(in out) :: x
  end subroutine
  subroutine intentinout_arr(x)
    real, intent(in out) :: x(:)
  end subroutine
  subroutine asynchronous(x)
    real, asynchronous :: x
  end subroutine
  subroutine asynchronous_arr(x)
    real, asynchronous :: x(:)
  end subroutine
  subroutine asynchronousValue(x)
    real, asynchronous, value :: x
  end subroutine
  subroutine volatile(x)
    real, volatile :: x
  end subroutine
  subroutine volatile_arr(x)
    real, volatile :: x(:)
  end subroutine
  subroutine pointer(x)
    real, pointer :: x(:)
  end subroutine
  subroutine valueassumedsize(x)
    real, intent(in) :: x(*)
  end subroutine
  subroutine volatileassumedsize(x)
    real, volatile :: x(*)
  end subroutine
  subroutine volatilecontiguous(x)
    real, volatile :: x(*)
  end subroutine

  subroutine test01(x) ! 15.5.2.4(2)
    class(t), intent(in) :: x[*]
    !ERROR: Coindexed polymorphic object may not be associated with a polymorphic dummy argument 'x='
    call poly(x[1])
  end subroutine

  subroutine mono(x)
    type(t), intent(in) :: x(*)
  end subroutine
  subroutine test02(x) ! 15.5.2.4(2)
    class(t), intent(in) :: x(*)
    !ERROR: Assumed-size polymorphic array may not be associated with a monomorphic dummy argument 'x='
    call mono(x)
  end subroutine

  subroutine typestar(x)
    type(*), intent(in) :: x
  end subroutine
  subroutine test03 ! 15.5.2.4(2)
    type(pdt(0)) :: x
    !ERROR: Actual argument associated with TYPE(*) dummy argument 'x=' may not have a parameterized derived type
    call typestar(x)
  end subroutine

  subroutine test04 ! 15.5.2.4(2)
    type(tbp) :: x
    !ERROR: Actual argument associated with TYPE(*) dummy argument 'x=' may not have type-bound procedure 'binding'
    call typestar(x)
  end subroutine

  subroutine test05 ! 15.5.2.4(2)
    type(final) :: x
    !ERROR: Actual argument associated with TYPE(*) dummy argument 'x=' may not have derived type 'final' with FINAL subroutine 'subr02'
    call typestar(x)
  end subroutine

  subroutine ch2(x)
    character(2), intent(in) :: x
  end subroutine
  subroutine pdtdefault (derivedArg)
    !ERROR: Type parameter 'n' lacks a value and has no default
    type(pdt) :: derivedArg
  end subroutine pdtdefault
  subroutine pdt3 (derivedArg)
    type(pdt(4)) :: derivedArg
  end subroutine pdt3
  subroutine pdt4 (derivedArg)
    type(pdt(*)) :: derivedArg
  end subroutine pdt4
  subroutine pdtWithDefaultDefault (derivedArg)
    type(pdtWithDefault) :: derivedArg
  end subroutine pdtWithDefaultdefault
  subroutine pdtWithDefault3 (derivedArg)
    type(pdtWithDefault(4)) :: derivedArg
  end subroutine pdtWithDefault3
  subroutine pdtWithDefault4 (derivedArg)
    type(pdtWithDefault(*)) :: derivedArg
  end subroutine pdtWithDefault4
  subroutine test06 ! 15.5.2.4(4)
    !ERROR: Type parameter 'n' lacks a value and has no default
    type(pdt) :: vardefault
    type(pdt(3)) :: var3
    type(pdt(4)) :: var4
    type(pdtWithDefault) :: defaultVardefault
    type(pdtWithDefault(3)) :: defaultVar3
    type(pdtWithDefault(4)) :: defaultVar4
    character :: ch1
    !ERROR: Actual argument variable length '1' is less than expected length '2'
    call ch2(ch1)
    !WARNING: Actual argument expression length '0' is less than expected length '2'
    call ch2("")
    call pdtdefault(vardefault)
    !ERROR: Actual argument type 'pdt(n=3_4)' is not compatible with dummy argument type 'pdt'
    call pdtdefault(var3)
    !ERROR: Actual argument type 'pdt(n=4_4)' is not compatible with dummy argument type 'pdt'
    call pdtdefault(var4) ! error
    !ERROR: Actual argument type 'pdt' is not compatible with dummy argument type 'pdt(n=4_4)'
    call pdt3(vardefault)
    !ERROR: Actual argument type 'pdt(n=3_4)' is not compatible with dummy argument type 'pdt(n=4_4)'
    call pdt3(var3)
    call pdt3(var4)
    !ERROR: Actual argument type 'pdt' is not compatible with dummy argument type 'pdt(n=*)'
    call pdt4(vardefault)
    call pdt4(var3)
    call pdt4(var4)
    call pdtWithDefaultdefault(defaultVardefault)
    call pdtWithDefaultdefault(defaultVar3)
    !ERROR: Actual argument type 'pdtwithdefault(n=4_4)' is not compatible with dummy argument type 'pdtwithdefault(n=3_4)'
    call pdtWithDefaultdefault(defaultVar4) ! error
    !ERROR: Actual argument type 'pdtwithdefault(n=3_4)' is not compatible with dummy argument type 'pdtwithdefault(n=4_4)'
    call pdtWithDefault3(defaultVardefault) ! error
    !ERROR: Actual argument type 'pdtwithdefault(n=3_4)' is not compatible with dummy argument type 'pdtwithdefault(n=4_4)'
    call pdtWithDefault3(defaultVar3) ! error
    call pdtWithDefault3(defaultVar4)
    call pdtWithDefault4(defaultVardefault)
    call pdtWithDefault4(defaultVar3)
    call pdtWithDefault4(defaultVar4)
  end subroutine

  subroutine out01(x)
    type(alloc) :: x
  end subroutine
  subroutine test07(x) ! 15.5.2.4(6)
    type(alloc) :: x[*]
    !ERROR: Coindexed actual argument with ALLOCATABLE ultimate component '%a' must be associated with a dummy argument 'x=' with VALUE or INTENT(IN) attributes
    call out01(x[1])
  end subroutine

  subroutine test08(x) ! 15.5.2.4(13)
    real :: x(1)[*]
    !ERROR: Coindexed scalar actual argument must be associated with a scalar dummy argument 'x='
    call assumedsize(x(1)[1])
  end subroutine

  subroutine charray(x)
    character :: x(10)
  end subroutine
  subroutine test09(ashape, polyarray, c, assumed_shape_char) ! 15.5.2.4(14), 15.5.2.11
    real :: x, arr(10)
    real, pointer :: p(:)
    real, pointer :: p_scalar
    character(10), pointer :: char_pointer(:)
    character(*) :: assumed_shape_char(:)
    real :: ashape(:)
    class(t) :: polyarray(*)
    character(10) :: c(:)
    !ERROR: Whole scalar actual argument may not be associated with a dummy argument 'x=' array
    call assumedsize(x)
    !ERROR: Whole scalar actual argument may not be associated with a dummy argument 'x=' array
    call assumedsize(p_scalar)
    !ERROR: Element of pointer array may not be associated with a dummy argument 'x=' array
    call assumedsize(p(1))
    !ERROR: Element of assumed-shape array may not be associated with a dummy argument 'x=' array
    call assumedsize(ashape(1))
    !ERROR: Polymorphic scalar may not be associated with a dummy argument 'x=' array
    call polyassumedsize(polyarray(1))
    call charray(c(1:1))  ! not an error if character
    call charray(char_pointer(1))  ! not an error if character
    call charray(assumed_shape_char(1))  ! not an error if character
    call assumedsize(arr(1))  ! not an error if element in sequence
    call assumedrank(x)  ! not an error
    call assumedtypeandsize(x)  ! not an error
  end subroutine

  subroutine test10(a) ! 15.5.2.4(16)
    real :: scalar, matrix(2,3)
    real :: a(*)
    !ERROR: Scalar actual argument may not be associated with assumed-shape dummy argument 'x='
    call assumedshape(scalar)
    call assumedshape(reshape(matrix,shape=[size(matrix)])) ! ok
    !ERROR: Rank of dummy argument is 1, but actual argument has rank 2
    call assumedshape(matrix)
    !ERROR: Assumed-size array may not be associated with assumed-shape dummy argument 'x='
    call assumedshape(a)
  end subroutine

  subroutine test11(in) ! C15.5.2.4(20)
    real, intent(in) :: in
    real :: x
    x = 0.
    !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'x=' is not definable
    !BECAUSE: 'in' is an INTENT(IN) dummy argument
    call intentout(in)
    !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'x=' is not definable
    !BECAUSE: '3.141590118408203125_4' is not a variable or pointer
    call intentout(3.14159)
    !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'x=' is not definable
    !BECAUSE: 'in+1._4' is not a variable or pointer
    call intentout(in + 1.)
    call intentout(x) ! ok
    !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'x=' is not definable
    !BECAUSE: '(x)' is not a variable or pointer
    call intentout((x))
    !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'count=' is not definable
    !BECAUSE: '2_4' is not a variable or pointer
    call system_clock(count=2)
    !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'x=' is not definable
    !BECAUSE: 'in' is an INTENT(IN) dummy argument
    call intentinout(in)
    !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'x=' is not definable
    !BECAUSE: '3.141590118408203125_4' is not a variable or pointer
    call intentinout(3.14159)
    !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'x=' is not definable
    !BECAUSE: 'in+1._4' is not a variable or pointer
    call intentinout(in + 1.)
    call intentinout(x) ! ok
    !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'x=' is not definable
    !BECAUSE: '(x)' is not a variable or pointer
    call intentinout((x))
    !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'exitstat=' is not definable
    !BECAUSE: '0_4' is not a variable or pointer
    call execute_command_line(command="echo hello", exitstat=0)
  end subroutine

  subroutine test12 ! 15.5.2.4(21)
    real :: a(1)
    integer :: j(1)
    j(1) = 1
    !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'x=' is not definable
    !BECAUSE: Variable 'a(int(j,kind=8))' has a vector subscript
    call intentout_arr(a(j))
    !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'x=' is not definable
    !BECAUSE: Variable 'a(int(j,kind=8))' has a vector subscript
    call intentinout_arr(a(j))
    call asynchronous_arr(a(j)) ! ok
    call volatile_arr(a(j)) ! ok
  end subroutine

  subroutine coarr(x)
    type(ultimateCoarray):: x
  end subroutine
  subroutine volcoarr(x)
    type(ultimateCoarray), volatile :: x
  end subroutine
  subroutine test13(a, b) ! 15.5.2.4(22)
    type(ultimateCoarray) :: a
    type(ultimateCoarray), volatile :: b
    call coarr(a)  ! ok
    call volcoarr(b)  ! ok
    !ERROR: VOLATILE attribute must match for dummy argument 'x=' when actual argument has a coarray ultimate component '%a'
    call coarr(b)
    !ERROR: VOLATILE attribute must match for dummy argument 'x=' when actual argument has a coarray ultimate component '%a'
    call volcoarr(a)
  end subroutine

  subroutine test14(a,b,c,d) ! C1538
    real :: a[*]
    real, asynchronous :: b[*]
    real, volatile :: c[*]
    real, asynchronous, volatile :: d[*]
    call asynchronous(a[1])  ! ok
    call volatile(a[1])  ! ok
    call asynchronousValue(b[1])  ! ok
    call asynchronousValue(c[1])  ! ok
    call asynchronousValue(d[1])  ! ok
    !ERROR: Coindexed ASYNCHRONOUS or VOLATILE actual argument may not be associated with dummy argument 'x=' with ASYNCHRONOUS or VOLATILE attributes unless VALUE
    call asynchronous(b[1])
    !ERROR: Coindexed ASYNCHRONOUS or VOLATILE actual argument may not be associated with dummy argument 'x=' with ASYNCHRONOUS or VOLATILE attributes unless VALUE
    call volatile(b[1])
    !ERROR: Coindexed ASYNCHRONOUS or VOLATILE actual argument may not be associated with dummy argument 'x=' with ASYNCHRONOUS or VOLATILE attributes unless VALUE
    call asynchronous(c[1])
    !ERROR: Coindexed ASYNCHRONOUS or VOLATILE actual argument may not be associated with dummy argument 'x=' with ASYNCHRONOUS or VOLATILE attributes unless VALUE
    call volatile(c[1])
    !ERROR: Coindexed ASYNCHRONOUS or VOLATILE actual argument may not be associated with dummy argument 'x=' with ASYNCHRONOUS or VOLATILE attributes unless VALUE
    call asynchronous(d[1])
    !ERROR: Coindexed ASYNCHRONOUS or VOLATILE actual argument may not be associated with dummy argument 'x=' with ASYNCHRONOUS or VOLATILE attributes unless VALUE
    call volatile(d[1])
  end subroutine

  subroutine test15(assumedrank) ! C1539
    real, pointer :: a(:)
    real, asynchronous :: b(10)
    real, volatile :: c(10)
    real, asynchronous, volatile :: d(10)
    real, asynchronous, volatile :: assumedrank(..)
    call assumedsize(a(::2)) ! ok
    call contiguous(a(::2)) ! ok
    call valueassumedsize(a(::2)) ! ok
    call valueassumedsize(b(::2)) ! ok
    call valueassumedsize(c(::2)) ! ok
    call valueassumedsize(d(::2)) ! ok
    !ERROR: ASYNCHRONOUS or VOLATILE actual argument that is not simply contiguous may not be associated with a contiguous ASYNCHRONOUS or VOLATILE dummy argument 'x='
    call volatileassumedsize(b(::2))
    !ERROR: ASYNCHRONOUS or VOLATILE actual argument that is not simply contiguous may not be associated with a contiguous ASYNCHRONOUS or VOLATILE dummy argument 'x='
    call volatilecontiguous(b(::2))
    !ERROR: ASYNCHRONOUS or VOLATILE actual argument that is not simply contiguous may not be associated with a contiguous ASYNCHRONOUS or VOLATILE dummy argument 'x='
    call volatileassumedsize(c(::2))
    !ERROR: ASYNCHRONOUS or VOLATILE actual argument that is not simply contiguous may not be associated with a contiguous ASYNCHRONOUS or VOLATILE dummy argument 'x='
    call volatilecontiguous(c(::2))
    !ERROR: ASYNCHRONOUS or VOLATILE actual argument that is not simply contiguous may not be associated with a contiguous ASYNCHRONOUS or VOLATILE dummy argument 'x='
    call volatileassumedsize(d(::2))
    !ERROR: ASYNCHRONOUS or VOLATILE actual argument that is not simply contiguous may not be associated with a contiguous ASYNCHRONOUS or VOLATILE dummy argument 'x='
    call volatilecontiguous(d(::2))
    !ERROR: ASYNCHRONOUS or VOLATILE actual argument that is not simply contiguous may not be associated with a contiguous ASYNCHRONOUS or VOLATILE dummy argument 'x='
    call volatilecontiguous(assumedrank)
  end subroutine

  subroutine test16() ! C1540
    real, pointer :: a(:)
    real, asynchronous, pointer :: b(:)
    real, volatile, pointer :: c(:)
    real, asynchronous, volatile, pointer :: d(:)
    call assumedsize(a) ! ok
    call contiguous(a) ! ok
    call pointer(a) ! ok
    call pointer(b) ! ok
    call pointer(c) ! ok
    call pointer(d) ! ok
    call valueassumedsize(a) ! ok
    call valueassumedsize(b) ! ok
    call valueassumedsize(c) ! ok
    call valueassumedsize(d) ! ok
    !ERROR: ASYNCHRONOUS or VOLATILE actual argument that is not simply contiguous may not be associated with a contiguous ASYNCHRONOUS or VOLATILE dummy argument 'x='
    call volatileassumedsize(b)
    !ERROR: ASYNCHRONOUS or VOLATILE actual argument that is not simply contiguous may not be associated with a contiguous ASYNCHRONOUS or VOLATILE dummy argument 'x='
    call volatilecontiguous(b)
    !ERROR: ASYNCHRONOUS or VOLATILE actual argument that is not simply contiguous may not be associated with a contiguous ASYNCHRONOUS or VOLATILE dummy argument 'x='
    call volatileassumedsize(c)
    !ERROR: ASYNCHRONOUS or VOLATILE actual argument that is not simply contiguous may not be associated with a contiguous ASYNCHRONOUS or VOLATILE dummy argument 'x='
    call volatilecontiguous(c)
    !ERROR: ASYNCHRONOUS or VOLATILE actual argument that is not simply contiguous may not be associated with a contiguous ASYNCHRONOUS or VOLATILE dummy argument 'x='
    call volatileassumedsize(d)
    !ERROR: ASYNCHRONOUS or VOLATILE actual argument that is not simply contiguous may not be associated with a contiguous ASYNCHRONOUS or VOLATILE dummy argument 'x='
    call volatilecontiguous(d)
  end subroutine

  subroutine explicitAsyncContig(x)
    real, asynchronous, intent(in out), contiguous :: x(:)
  end
  subroutine implicitAsyncContig(x)
    real, intent(in out), contiguous :: x(:)
    read(1,id=id,asynchronous="yes") x
  end
  subroutine test17explicit(x)
    real, asynchronous, intent(in out) :: x(:)
    !ERROR: ASYNCHRONOUS or VOLATILE actual argument that is not simply contiguous may not be associated with a contiguous ASYNCHRONOUS or VOLATILE dummy argument 'x='
    call explicitAsyncContig(x)
    !ERROR: ASYNCHRONOUS or VOLATILE actual argument that is not simply contiguous may not be associated with a contiguous ASYNCHRONOUS or VOLATILE dummy argument 'x='
    call implicitAsyncContig(x)
  end
  subroutine test17implicit(x)
    real, intent(in out) :: x(:)
    read(1,id=id,asynchronous="yes") x
    !ERROR: ASYNCHRONOUS or VOLATILE actual argument that is not simply contiguous may not be associated with a contiguous ASYNCHRONOUS or VOLATILE dummy argument 'x='
    call explicitAsyncContig(x)
    !ERROR: ASYNCHRONOUS or VOLATILE actual argument that is not simply contiguous may not be associated with a contiguous ASYNCHRONOUS or VOLATILE dummy argument 'x='
    call implicitAsyncContig(x)
  end
  subroutine test17block(x)
    real, intent(in out) :: x(:)
    call explicitAsyncContig(x) ! ok
    call implicitAsyncContig(x) ! ok
    block
      read(1,id=id,asynchronous="yes") x
      !ERROR: ASYNCHRONOUS or VOLATILE actual argument that is not simply contiguous may not be associated with a contiguous ASYNCHRONOUS or VOLATILE dummy argument 'x='
      call explicitAsyncContig(x)
      !ERROR: ASYNCHRONOUS or VOLATILE actual argument that is not simply contiguous may not be associated with a contiguous ASYNCHRONOUS or VOLATILE dummy argument 'x='
      call implicitAsyncContig(x)
    end block
  end
  subroutine test17internal(x)
    real, intent(in out) :: x(:)
    call explicitAsyncContig(x) ! ok
    call implicitAsyncContig(x) ! ok
   contains
    subroutine internal
      read(1,id=id,asynchronous="yes") x
      !ERROR: ASYNCHRONOUS or VOLATILE actual argument that is not simply contiguous may not be associated with a contiguous ASYNCHRONOUS or VOLATILE dummy argument 'x='
      call explicitAsyncContig(x)
      !ERROR: ASYNCHRONOUS or VOLATILE actual argument that is not simply contiguous may not be associated with a contiguous ASYNCHRONOUS or VOLATILE dummy argument 'x='
      call implicitAsyncContig(x)
    end
  end

end module

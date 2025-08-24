! RUN: not %flang_fc1 %s 2>&1 | FileCheck %s
! Check errors found in folding
! TODO: test others emitted from flang/lib/Evaluate
module m
  type t
    real x
  end type t
 contains
  subroutine s1(a,b,c,d)
    real :: a(*), b(:), c(..), d
    !CHECK: error: DIM=1 dimension is out of range for rank-1 assumed-size array
    integer :: ub1(ubound(a,1))
    !CHECK-NOT: error: DIM=1 dimension is out of range for rank-1 assumed-size array
    integer :: lb1(lbound(a,1))
    !CHECK: error: DIM=0 dimension must be positive
    integer :: ub2(ubound(a,0))
    !CHECK: error: DIM=2 dimension is out of range for rank-1 array
    integer :: ub3(ubound(a,2))
    !CHECK: error: DIM=0 dimension must be positive
    integer :: lb2(lbound(b,0))
    !CHECK: error: DIM=2 dimension is out of range for rank-1 array
    integer :: lb3(lbound(b,2))
    !CHECK: error: DIM=0 dimension must be positive
    integer :: lb4(lbound(c,0))
    !CHECK: error: DIM=666 dimension is too large for any array (maximum rank 15)
    integer :: lb5(lbound(c,666))
    !CHECK: error: 'array=' argument has unacceptable rank 0
    integer :: lb6(lbound(d,1))
    !CHECK: error: 'array=' argument has unacceptable rank 0
    integer :: ub4(ubound(d,1))
  end subroutine
  subroutine s2
    integer, parameter :: array(2,3) = reshape([(j, j=1, 6)], shape(array))
    integer :: x(2, 3)
    !CHECK: error: Invalid 'dim=' argument (0) in CSHIFT
    x = cshift(array, [1, 2], dim=0)
    !CHECK: error: Invalid 'shift=' argument in CSHIFT: extent on dimension 1 is 2 but must be 3
    x = cshift(array, [1, 2], dim=1)
  end subroutine
  subroutine s3
    integer, parameter :: array(2,3) = reshape([(j, j=1, 6)], shape(array))
    integer :: x(2, 3)
    !CHECK: error: Invalid 'dim=' argument (0) in EOSHIFT
    x = eoshift(array, [1, 2], dim=0)
    !CHECK: error: Invalid 'shift=' argument in EOSHIFT: extent on dimension 1 is 2 but must be 3
    x = eoshift(array, [1, 2], dim=1)
    !CHECK: error: Invalid 'boundary=' argument in EOSHIFT: extent on dimension 1 is 3 but must be 2
    x = eoshift(array, 1, [0, 0, 0], 2)
  end subroutine
  subroutine s4
    integer, parameter :: array(2,3) = reshape([(j, j=1, 6)], shape(array))
    logical, parameter :: mask(*,*) = reshape([(.true., j=1,3),(.false., j=1,3)], shape(array))
    integer :: x(3)
    !CHECK: error: Invalid 'vector=' argument in PACK: the 'mask=' argument has 3 true elements, but the vector has only 2 elements
    x = pack(array, mask, [0,0])
    x = pack(spread(array, x(1), x(2)), .true.) ! regression check, once crashed
  end subroutine
  subroutine s5
    logical, parameter :: mask(2,3) = reshape([.false., .true., .true., .false., .false., .true.], shape(mask))
    integer, parameter :: field(3,2) = reshape([(-j,j=1,6)], shape(field))
    integer :: x(2,3)
    !CHECK: error: Invalid 'vector=' argument in UNPACK: the 'mask=' argument has 3 true elements, but the vector has only 2 elements
    x = unpack([1,2], mask, 0)
  end subroutine
  subroutine s6
    !CHECK: error: POS=-1 out of range for BTEST
    logical, parameter :: bad1 = btest(0, -1)
    !CHECK: error: POS=32 out of range for BTEST
    logical, parameter :: bad2 = btest(0, 32)
    !CHECK-NOT: error: POS=33 out of range for BTEST
    logical, parameter :: ok1 = btest(0_8, 33)
    !CHECK: error: POS=64 out of range for BTEST
    logical, parameter :: bad4 = btest(0_8, 64)
  end subroutine
  subroutine s7
    !CHECK: error: SHIFT=-33 count for ishft is less than -32
    integer, parameter :: bad1 = ishft(1, -33)
    integer, parameter :: ok1 = ishft(1, -32)
    integer, parameter :: ok2 = ishft(1, 32)
    !CHECK: error: SHIFT=33 count for ishft is greater than 32
    integer, parameter :: bad2 = ishft(1, 33)
    !CHECK: error: SHIFT=-65 count for ishft is less than -64
    integer(8), parameter :: bad3 = ishft(1_8, -65)
    integer(8), parameter :: ok3 = ishft(1_8, -64)
    integer(8), parameter :: ok4 = ishft(1_8, 64)
    !CHECK: error: SHIFT=65 count for ishft is greater than 64
    integer(8), parameter :: bad4 = ishft(1_8, 65)
  end subroutine
  subroutine s8
    !CHECK: error: SHIFT=-33 count for shiftl is negative
    integer, parameter :: bad1 = shiftl(1, -33)
    !CHECK: error: SHIFT=-32 count for shiftl is negative
    integer, parameter :: bad2 = shiftl(1, -32)
    integer, parameter :: ok1 = shiftl(1, 32)
    !CHECK: error: SHIFT=33 count for shiftl is greater than 32
    integer, parameter :: bad3 = shiftl(1, 33)
    !CHECK: error: SHIFT=-65 count for shiftl is negative
    integer(8), parameter :: bad4 = shiftl(1_8, -65)
    !CHECK: error: SHIFT=-64 count for shiftl is negative
    integer(8), parameter :: bad5 = shiftl(1_8, -64)
    integer(8), parameter :: ok2 = shiftl(1_8, 64)
    !CHECK: error: SHIFT=65 count for shiftl is greater than 64
    integer(8), parameter :: bad6 = shiftl(1_8, 65)
  end subroutine
  subroutine s9
    integer, parameter :: rank15(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1) = 1
    !CHECK: error: SOURCE= argument to SPREAD has rank 15 but must have rank less than 15
    integer, parameter :: bad1 = spread(rank15, 1, 1)
    integer, parameter :: matrix(2, 2) = reshape([1, 2, 3, 4], [2, 2])
    !CHECK: error: DIM=0 argument to SPREAD must be between 1 and 3
    integer, parameter :: bad2 = spread(matrix, 0, 1)
    !CHECK: error: DIM=4 argument to SPREAD must be between 1 and 3
    integer, parameter :: bad3 = spread(matrix, 4, 1)
  end subroutine
  subroutine s10
    !CHECK: warning: CHAR(I=-1) is out of range for CHARACTER(KIND=1)
    character(kind=1), parameter :: badc11 = char(-1,kind=1)
    !CHECK: warning: ACHAR(I=-1) is out of range for CHARACTER(KIND=1)
    character(kind=1), parameter :: bada11 = achar(-1,kind=1)
    !CHECK: warning: CHAR(I=-1) is out of range for CHARACTER(KIND=2)
    character(kind=2), parameter :: badc21 = char(-1,kind=2)
    !CHECK: warning: ACHAR(I=-1) is out of range for CHARACTER(KIND=2)
    character(kind=2), parameter :: bada21 = achar(-1,kind=2)
    !CHECK: warning: CHAR(I=-1) is out of range for CHARACTER(KIND=4)
    character(kind=4), parameter :: badc41 = char(-1,kind=4)
    !CHECK: warning: ACHAR(I=-1) is out of range for CHARACTER(KIND=4)
    character(kind=4), parameter :: bada41 = achar(-1,kind=4)
    !CHECK: warning: CHAR(I=256) is out of range for CHARACTER(KIND=1)
    character(kind=1), parameter :: badc12 = char(256,kind=1)
    !CHECK: warning: ACHAR(I=256) is out of range for CHARACTER(KIND=1)
    character(kind=1), parameter :: bada12 = achar(256,kind=1)
    !CHECK: warning: CHAR(I=65536) is out of range for CHARACTER(KIND=2)
    character(kind=2), parameter :: badc22 = char(65536,kind=2)
    !CHECK: warning: ACHAR(I=65536) is out of range for CHARACTER(KIND=2)
    character(kind=2), parameter :: bada22 = achar(65536,kind=2)
    !CHECK: warning: CHAR(I=4294967296) is out of range for CHARACTER(KIND=4)
    character(kind=4), parameter :: badc42 = char(4294967296_8,kind=4)
    !CHECK: warning: ACHAR(I=4294967296) is out of range for CHARACTER(KIND=4)
    character(kind=4), parameter :: bada42 = achar(4294967296_8,kind=4)
  end subroutine
  subroutine s11
    character(:), allocatable :: x1
    !CHECK: error: Invalid specification expression: non-constant inquiry function 'len' not allowed for local object
    character(len(x1)) :: x2
    real, allocatable :: x3(:)
    !CHECK: error: Invalid specification expression: non-constant descriptor inquiry not allowed for local object
    real :: x4(size(x3))
  end
  subroutine s12(x,y)
    class(t), intent(in) :: x
    class(*), intent(in) :: y
    !CHECK: error: Must be a constant value
    integer, parameter :: bad1 = storage_size(x)
    !CHECK: error: Must be a constant value
    integer, parameter :: bad2 = storage_size(y)
  end subroutine
  subroutine s13
    !CHECK: portability: Result of REPEAT() is too large to compute at compilation time (1.1259e+15 characters)
    print *, repeat(repeat(' ', 2**20), 2**30)
    !CHECK: error: NCOPIES= argument to REPEAT() should be nonnegative, but is -666
    print *, repeat(' ', -666)
  end subroutine
  subroutine s14(n)
    integer, intent(in) :: n
    !CHECK: error: bit position for IBITS(POS=-1) is negative
    print *, ibits(0, -1, n)
    !CHECK: error: bit length for IBITS(LEN=-1) is negative
    print *, ibits(0, n, -1)
    !CHECK: error: IBITS() must have POS+LEN (>=33) no greater than 32
    print *, ibits(0, n, 33)
    !CHECK: error: IBITS() must have POS+LEN (>=33) no greater than 32
    print *, ibits(0, 33, n)
  end
  subroutine s15
    use ieee_arithmetic, only: ieee_flag_type, ieee_underflow, ieee_support_flag
    type(ieee_flag_type) :: f1 = ieee_underflow, f2
    !CHECK: portability: specification expression refers to local object 'f1' (initialized and saved)
    integer ok(merge(kind(1),-1,ieee_support_flag(f1, x)))
    !CHECK: error: Invalid specification expression: reference to local entity 'f2'
    integer bad(merge(kind(1),-1,ieee_support_flag(f2, x)))
  end
  subroutine warnings
    use ieee_arithmetic, only: ieee_scalb
    real, parameter :: ok1 = scale(0.0, 99999) ! 0.0
    real, parameter :: ok2 = scale(1.0, -99999) ! 0.0
    !CHECK: SCALE/IEEE_SCALB intrinsic folding overflow
    real, parameter :: bad1 = scale(1.0, 99999)
    !CHECK: SCALE/IEEE_SCALB intrinsic folding overflow
    real, parameter :: bad1a = ieee_scalb(1.0, 99999)
    !CHECK: complex ABS intrinsic folding overflow
    real, parameter :: bad2 = abs(cmplx(huge(0.),huge(0.)))
    !CHECK: warning: DIM intrinsic folding overflow
    real, parameter :: bad3 = dim(huge(1.),-.5*huge(1.))
    !CHECK: warning: DIM intrinsic folding overflow
    integer, parameter :: bad4 = dim(huge(1),-1)
    !CHECK: warning: HYPOT intrinsic folding overflow
    real, parameter :: bad5 = hypot(huge(0.), huge(0.))
    !CHECK: warning: SUM() of INTEGER(4) data overflowed
    integer, parameter :: bad6 = sum([huge(1),huge(1)])
    !CHECK: warning: SUM() of REAL(4) data overflowed
    real, parameter :: bad7 = sum([huge(1.),huge(1.)])
    !CHECK: warning: SUM() of COMPLEX(4) data overflowed
    complex, parameter :: bad8 = sum([(huge(1.),0.),(huge(1.),0.)])
    !CHECK: warning: PRODUCT() of INTEGER(4) data overflowed
    integer, parameter :: bad9 = product([huge(1),huge(1)])
    !CHECK: warning: PRODUCT() of REAL(4) data overflowed
    real, parameter :: bad10 = product([huge(1.),huge(1.)])
    !CHECK: warning: PRODUCT() of COMPLEX(4) data overflowed
    complex, parameter :: bad11 = product([(huge(1.),0.),(huge(1.),0.)])
    !CHECK: warning: conversion of 111111111111111111111_16 to INTEGER(8) overflowed; result is 430646668853801415
    integer(8), parameter :: bad12 = int(111111111111111111111, 8)
    !CHECK: warning: overflow on REAL(8) to REAL(4) conversion
    x = 1.D40
    !CHECK-NOT: warning: invalid argument
    if (.not. isnan(real(z'ffffffffffffffff',8))) stop
  end subroutine
end module

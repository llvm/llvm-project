! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic -Werror
! Tests the checking of storage sequence argument association (F'2023 15.2.5.12)
module nonchar
 contains
  subroutine scalar(a)
    real a
  end
  subroutine explicit1(a)
    real a(2)
  end
  subroutine explicit2(a)
    real a(2,2)
  end
  subroutine assumedSize1(a)
    real a(*)
  end
  subroutine assumedSize2(a)
    real a(2,*)
  end
  subroutine assumedShape1(a)
    real a(:)
  end
  subroutine assumedShape2(a)
    real a(:,:)
  end
  subroutine assumedRank(a)
    real a(..)
  end
  subroutine allocatable0(a)
    real, allocatable :: a
  end
  subroutine allocatable1(a)
    real, allocatable :: a(:)
  end
  subroutine allocatable2(a)
    real, allocatable :: a(:,:)
  end
  subroutine pointer0(a)
    real, intent(in), pointer :: a
  end
  subroutine pointer1(a)
    real, intent(in), pointer :: a(:)
  end
  subroutine pointer2(a)
    real, intent(in), pointer :: a(:,:)
  end
  subroutine coarray0(a)
    real a[*]
  end

  subroutine test
    real, target :: scalar0
    real, target :: vector1(1), vector2(2), vector4(4)
    real, target ::  matrix11(1,1), matrix12(1,2), matrix22(2,2)
    real, allocatable :: alloScalar, alloVector(:), alloMatrix(:,:)

    call scalar(scalar0)
    !ERROR: Rank of dummy argument is 0, but actual argument has rank 1
    call scalar(vector1)
    call scalar(vector1(1))

    !ERROR: Whole scalar actual argument may not be associated with a dummy argument 'a=' array
    call explicit1(scalar0)
    !ERROR: Actual argument array has fewer elements (1) than dummy argument 'a=' array (2)
    call explicit1(vector1)
    call explicit1(vector2)
    call explicit1(vector4)
    !ERROR: Actual argument has fewer elements remaining in storage sequence (1) than dummy argument 'a=' array (2)
    call explicit1(vector2(2))
    call explicit1(vector4(3))
    !ERROR: Actual argument has fewer elements remaining in storage sequence (1) than dummy argument 'a=' array (2)
    call explicit1(vector4(4))
    !ERROR: Actual argument array has fewer elements (1) than dummy argument 'a=' array (2)
    call explicit1(matrix11)

    !ERROR: Whole scalar actual argument may not be associated with a dummy argument 'a=' array
    call explicit2(scalar0)
    !ERROR: Actual argument array has fewer elements (1) than dummy argument 'a=' array (4)
    call explicit2(vector1)
    !ERROR: Actual argument array has fewer elements (2) than dummy argument 'a=' array (4)
    call explicit2(vector2)
    call explicit2(vector4)
    !ERROR: Actual argument has fewer elements remaining in storage sequence (1) than dummy argument 'a=' array (4)
    call explicit2(vector2(2))
    !ERROR: Actual argument has fewer elements remaining in storage sequence (3) than dummy argument 'a=' array (4)
    call explicit2(vector4(2))
    call explicit2(vector4(1))
    !ERROR: Actual argument array has fewer elements (1) than dummy argument 'a=' array (4)
    call explicit2(matrix11)
    !ERROR: Actual argument array has fewer elements (2) than dummy argument 'a=' array (4)
    call explicit2(matrix12)
    call explicit2(matrix22)
    call explicit2(matrix22(1,1))
    !ERROR: Actual argument has fewer elements remaining in storage sequence (3) than dummy argument 'a=' array (4)
    call explicit2(matrix22(2,1))

    !ERROR: Whole scalar actual argument may not be associated with a dummy argument 'a=' array
    call assumedSize1(scalar0)
    call assumedSize1(vector1)
    call assumedSize1(vector2)
    call assumedSize1(vector4)
    call assumedSize1(vector2(2))
    call assumedSize1(vector4(2))
    call assumedSize1(vector4(1))
    call assumedSize1(matrix11)
    call assumedSize1(matrix12)
    call assumedSize1(matrix22)
    call assumedSize1(matrix22(1,1))
    call assumedSize1(matrix22(2,1))

    !ERROR: Whole scalar actual argument may not be associated with a dummy argument 'a=' array
    call assumedSize2(scalar0)
    call assumedSize2(vector1)
    call assumedSize2(vector2)
    call assumedSize2(vector4)
    call assumedSize2(vector2(2))
    call assumedSize2(vector4(2))
    call assumedSize2(vector4(1))
    call assumedSize2(matrix11)
    call assumedSize2(matrix12)
    call assumedSize2(matrix22)
    call assumedSize2(matrix22(1,1))
    call assumedSize2(matrix22(2,1))

    !ERROR: Scalar actual argument may not be associated with assumed-shape dummy argument 'a='
    call assumedShape1(scalar0)
    call assumedShape1(vector1)
    call assumedShape1(vector2)
    call assumedShape1(vector4)
    !ERROR: Scalar actual argument may not be associated with assumed-shape dummy argument 'a='
    call assumedShape1(vector2(2))
    !ERROR: Rank of dummy argument is 1, but actual argument has rank 2
    call assumedShape1(matrix11)
    !ERROR: Rank of dummy argument is 1, but actual argument has rank 2
    call assumedShape1(matrix12)
    !ERROR: Rank of dummy argument is 1, but actual argument has rank 2
    call assumedShape1(matrix22)
    !ERROR: Scalar actual argument may not be associated with assumed-shape dummy argument 'a='
    call assumedShape1(matrix22(1,1))

    !ERROR: Scalar actual argument may not be associated with assumed-shape dummy argument 'a='
    call assumedShape2(scalar0)
    !ERROR: Rank of dummy argument is 2, but actual argument has rank 1
    call assumedShape2(vector1)
    !ERROR: Rank of dummy argument is 2, but actual argument has rank 1
    call assumedShape2(vector2)
    !ERROR: Rank of dummy argument is 2, but actual argument has rank 1
    call assumedShape2(vector4)
    !ERROR: Scalar actual argument may not be associated with assumed-shape dummy argument 'a='
    call assumedShape2(vector2(2))
    call assumedShape2(matrix11)
    call assumedShape2(matrix12)
    call assumedShape2(matrix22)
    !ERROR: Scalar actual argument may not be associated with assumed-shape dummy argument 'a='
    call assumedShape2(matrix22(1,1))

    call assumedRank(scalar0)
    call assumedRank(vector1)
    call assumedRank(vector1(1))
    call assumedRank(matrix11)
    call assumedRank(matrix11(1,1))

    !ERROR: ALLOCATABLE dummy argument 'a=' must be associated with an ALLOCATABLE actual argument
    call allocatable0(scalar0)
    call allocatable0(alloScalar)
    !ERROR: Rank of dummy argument is 0, but actual argument has rank 1
    call allocatable0(alloVector)
    !ERROR: ALLOCATABLE dummy argument 'a=' must be associated with an ALLOCATABLE actual argument
    call allocatable0(alloVector(1))
    !ERROR: Rank of dummy argument is 0, but actual argument has rank 2
    call allocatable0(alloMatrix)
    !ERROR: ALLOCATABLE dummy argument 'a=' must be associated with an ALLOCATABLE actual argument
    call allocatable0(alloMatrix(1,1))

    !ERROR: Rank of dummy argument is 1, but actual argument has rank 0
    !ERROR: ALLOCATABLE dummy argument 'a=' must be associated with an ALLOCATABLE actual argument
    call allocatable1(scalar0)
    !ERROR: Rank of dummy argument is 1, but actual argument has rank 0
    call allocatable1(alloScalar)
    call allocatable1(alloVector)
    !ERROR: Rank of dummy argument is 1, but actual argument has rank 0
    !ERROR: ALLOCATABLE dummy argument 'a=' must be associated with an ALLOCATABLE actual argument
    call allocatable1(alloVector(1))
    !ERROR: Rank of dummy argument is 1, but actual argument has rank 2
    call allocatable1(alloMatrix)
    !ERROR: Rank of dummy argument is 1, but actual argument has rank 0
    !ERROR: ALLOCATABLE dummy argument 'a=' must be associated with an ALLOCATABLE actual argument
    call allocatable1(alloMatrix(1,1))

    !ERROR: Rank of dummy argument is 2, but actual argument has rank 0
    !ERROR: ALLOCATABLE dummy argument 'a=' must be associated with an ALLOCATABLE actual argument
    call allocatable2(scalar0)
    !ERROR: Rank of dummy argument is 2, but actual argument has rank 0
    call allocatable2(alloScalar)
    !ERROR: Rank of dummy argument is 2, but actual argument has rank 1
    call allocatable2(alloVector)
    !ERROR: Rank of dummy argument is 2, but actual argument has rank 0
    !ERROR: ALLOCATABLE dummy argument 'a=' must be associated with an ALLOCATABLE actual argument
    call allocatable2(alloVector(1))
    call allocatable2(alloMatrix)
    !ERROR: Rank of dummy argument is 2, but actual argument has rank 0
    !ERROR: ALLOCATABLE dummy argument 'a=' must be associated with an ALLOCATABLE actual argument
    call allocatable2(alloMatrix(1,1))

    call pointer0(scalar0)
    !ERROR: Rank of dummy argument is 0, but actual argument has rank 1
    !ERROR: Pointer has rank 0 but target has rank 1
    call pointer0(vector1)
    call pointer0(vector1(1))
    !ERROR: Rank of dummy argument is 0, but actual argument has rank 2
    !ERROR: Pointer has rank 0 but target has rank 2
    call pointer0(matrix11)
    call pointer0(matrix11(1,1))

    !ERROR: Rank of dummy argument is 1, but actual argument has rank 0
    !ERROR: Pointer has rank 1 but target has rank 0
    call pointer1(scalar0)
    call pointer1(vector1)
    !ERROR: Rank of dummy argument is 1, but actual argument has rank 0
    !ERROR: Pointer has rank 1 but target has rank 0
    call pointer1(vector1(1))
    !ERROR: Rank of dummy argument is 1, but actual argument has rank 2
    !ERROR: Pointer has rank 1 but target has rank 2
    call pointer1(matrix11)
    !ERROR: Rank of dummy argument is 1, but actual argument has rank 0
    !ERROR: Pointer has rank 1 but target has rank 0
    call pointer1(matrix11(1,1))

    !ERROR: Rank of dummy argument is 2, but actual argument has rank 0
    !ERROR: Pointer has rank 2 but target has rank 0
    call pointer2(scalar0)
    !ERROR: Rank of dummy argument is 2, but actual argument has rank 1
    !ERROR: Pointer has rank 2 but target has rank 1
    call pointer2(vector1)
    !ERROR: Rank of dummy argument is 2, but actual argument has rank 0
    !ERROR: Pointer has rank 2 but target has rank 0
    call pointer2(vector1(1))
    call pointer2(matrix11)
    !ERROR: Rank of dummy argument is 2, but actual argument has rank 0
    !ERROR: Pointer has rank 2 but target has rank 0
    call pointer2(matrix11(1,1))

    !ERROR: Actual argument associated with coarray dummy argument 'a=' must be a coarray
    call coarray0(scalar0)
    !ERROR: Rank of dummy argument is 0, but actual argument has rank 1
    !ERROR: Actual argument associated with coarray dummy argument 'a=' must be a coarray
    call coarray0(vector1)
    !ERROR: Actual argument associated with coarray dummy argument 'a=' must be a coarray
    call coarray0(vector1(1))
    !ERROR: Rank of dummy argument is 0, but actual argument has rank 2
    !ERROR: Actual argument associated with coarray dummy argument 'a=' must be a coarray
    call coarray0(matrix11)
    !ERROR: Actual argument associated with coarray dummy argument 'a=' must be a coarray
    call coarray0(matrix11(1,1))
  end
end

module char
 contains
  subroutine scalar(a)
    character(2) a
  end
  subroutine explicit1(a)
    character(2) a(2)
  end
  subroutine explicit2(a)
    character(2) a(2,2)
  end
  subroutine assumedSize1(a)
    character(2) a(*)
  end
  subroutine assumedSize2(a)
    character(2) a(2,*)
  end
  subroutine assumedShape1(a)
    character(2) a(:)
  end
  subroutine assumedShape2(a)
    character(2) a(:,:)
  end
  subroutine assumedRank(a)
    character(2) a(..)
  end
  subroutine allocatable0(a)
    character(2), allocatable :: a
  end
  subroutine allocatable1(a)
    character(2), allocatable :: a(:)
  end
  subroutine allocatable2(a)
    character(2), allocatable :: a(:,:)
  end
  subroutine pointer0(a)
    character(2), intent(in), pointer :: a
  end
  subroutine pointer1(a)
    character(2), intent(in), pointer :: a(:)
  end
  subroutine pointer2(a)
    character(2), intent(in), pointer :: a(:,:)
  end
  subroutine coarray0(a)
    character(2) a[*]
  end

  subroutine test
    character(2), target :: scalar0
    character(2), target :: vector1(1), vector2(2), vector4(4)
    character(2), target ::  matrix11(1,1), matrix12(1,2), matrix22(2,2)
    character(2), allocatable :: alloScalar, alloVector(:), alloMatrix(:,:)

    call scalar(scalar0)
    !ERROR: Rank of dummy argument is 0, but actual argument has rank 1
    call scalar(vector1)
    call scalar(vector1(1))

    !ERROR: Actual argument has fewer characters remaining in storage sequence (2) than dummy argument 'a=' (4)
    call explicit1(scalar0)
    !ERROR: Actual argument array has fewer characters (2) than dummy argument 'a=' array (4)
    call explicit1(vector1)
    call explicit1(vector2)
    call explicit1(vector4)
    !ERROR: Actual argument has fewer characters remaining in storage sequence (2) than dummy argument 'a=' (4)
    call explicit1(vector2(2))
    !ERROR: Actual argument has fewer characters remaining in storage sequence (3) than dummy argument 'a=' (4)
    call explicit1(vector2(1)(2:2))
    call explicit1(vector4(3))
    !ERROR: Actual argument has fewer characters remaining in storage sequence (2) than dummy argument 'a=' (4)
    call explicit1(vector4(4))
    !ERROR: Actual argument array has fewer characters (2) than dummy argument 'a=' array (4)
    call explicit1(matrix11)
    call explicit1(matrix12)
    call explicit1(matrix12(1,1))
    !ERROR: Actual argument has fewer characters remaining in storage sequence (3) than dummy argument 'a=' (4)
    call explicit1(matrix12(1,1)(2:2))
    !ERROR: Actual argument has fewer characters remaining in storage sequence (2) than dummy argument 'a=' (4)
    call explicit1(matrix12(1,2))

    !ERROR: Actual argument has fewer characters remaining in storage sequence (2) than dummy argument 'a=' (8)
    call explicit2(scalar0)
    !ERROR: Actual argument array has fewer characters (2) than dummy argument 'a=' array (8)
    call explicit2(vector1)
    !ERROR: Actual argument array has fewer characters (4) than dummy argument 'a=' array (8)
    call explicit2(vector2)
    call explicit2(vector4)
    !ERROR: Actual argument has fewer characters remaining in storage sequence (2) than dummy argument 'a=' (8)
    call explicit2(vector2(2))
    !ERROR: Actual argument has fewer characters remaining in storage sequence (6) than dummy argument 'a=' (8)
    call explicit2(vector4(2))
    call explicit2(vector4(1))
    !ERROR: Actual argument array has fewer characters (2) than dummy argument 'a=' array (8)
    call explicit2(matrix11)
    !ERROR: Actual argument array has fewer characters (4) than dummy argument 'a=' array (8)
    call explicit2(matrix12)
    call explicit2(matrix22)
    call explicit2(matrix22(1,1))
    !ERROR: Actual argument has fewer characters remaining in storage sequence (7) than dummy argument 'a=' (8)
    call explicit2(matrix22(1,1)(2:2))
    !ERROR: Actual argument has fewer characters remaining in storage sequence (6) than dummy argument 'a=' (8)
    call explicit2(matrix22(2,1))

    call assumedSize1(scalar0)
    call assumedSize1(vector1)
    call assumedSize1(vector2)
    call assumedSize1(vector4)
    call assumedSize1(vector2(2))
    call assumedSize1(vector4(2))
    call assumedSize1(vector4(1))
    call assumedSize1(matrix11)
    call assumedSize1(matrix12)
    call assumedSize1(matrix22)
    call assumedSize1(matrix22(1,1))
    call assumedSize1(matrix22(2,1))

    call assumedSize2(scalar0)
    call assumedSize2(vector1)
    call assumedSize2(vector2)
    call assumedSize2(vector4)
    call assumedSize2(vector2(2))
    call assumedSize2(vector4(2))
    call assumedSize2(vector4(1))
    call assumedSize2(matrix11)
    call assumedSize2(matrix12)
    call assumedSize2(matrix22)
    call assumedSize2(matrix22(1,1))
    call assumedSize2(matrix22(2,1))

    !ERROR: Scalar actual argument may not be associated with assumed-shape dummy argument 'a='
    call assumedShape1(scalar0)
    call assumedShape1(vector1)
    call assumedShape1(vector2)
    call assumedShape1(vector4)
    !ERROR: Scalar actual argument may not be associated with assumed-shape dummy argument 'a='
    call assumedShape1(vector2(2))
    !ERROR: Rank of dummy argument is 1, but actual argument has rank 2
    call assumedShape1(matrix11)
    !ERROR: Rank of dummy argument is 1, but actual argument has rank 2
    call assumedShape1(matrix12)
    !ERROR: Rank of dummy argument is 1, but actual argument has rank 2
    call assumedShape1(matrix22)
    !ERROR: Scalar actual argument may not be associated with assumed-shape dummy argument 'a='
    call assumedShape1(matrix22(1,1))

    !ERROR: Scalar actual argument may not be associated with assumed-shape dummy argument 'a='
    call assumedShape2(scalar0)
    !ERROR: Rank of dummy argument is 2, but actual argument has rank 1
    call assumedShape2(vector1)
    !ERROR: Rank of dummy argument is 2, but actual argument has rank 1
    call assumedShape2(vector2)
    !ERROR: Rank of dummy argument is 2, but actual argument has rank 1
    call assumedShape2(vector4)
    !ERROR: Scalar actual argument may not be associated with assumed-shape dummy argument 'a='
    call assumedShape2(vector2(2))
    call assumedShape2(matrix11)
    call assumedShape2(matrix12)
    call assumedShape2(matrix22)
    !ERROR: Scalar actual argument may not be associated with assumed-shape dummy argument 'a='
    call assumedShape2(matrix22(1,1))

    call assumedRank(scalar0)
    call assumedRank(vector1)
    call assumedRank(vector1(1))
    call assumedRank(matrix11)
    call assumedRank(matrix11(1,1))

    !ERROR: ALLOCATABLE dummy argument 'a=' must be associated with an ALLOCATABLE actual argument
    call allocatable0(scalar0)
    call allocatable0(alloScalar)
    !ERROR: Rank of dummy argument is 0, but actual argument has rank 1
    call allocatable0(alloVector)
    !ERROR: ALLOCATABLE dummy argument 'a=' must be associated with an ALLOCATABLE actual argument
    call allocatable0(alloVector(1))
    !ERROR: Rank of dummy argument is 0, but actual argument has rank 2
    call allocatable0(alloMatrix)
    !ERROR: ALLOCATABLE dummy argument 'a=' must be associated with an ALLOCATABLE actual argument
    call allocatable0(alloMatrix(1,1))

    !ERROR: Rank of dummy argument is 1, but actual argument has rank 0
    !ERROR: ALLOCATABLE dummy argument 'a=' must be associated with an ALLOCATABLE actual argument
    call allocatable1(scalar0)
    !ERROR: Rank of dummy argument is 1, but actual argument has rank 0
    call allocatable1(alloScalar)
    call allocatable1(alloVector)
    !ERROR: Rank of dummy argument is 1, but actual argument has rank 0
    !ERROR: ALLOCATABLE dummy argument 'a=' must be associated with an ALLOCATABLE actual argument
    call allocatable1(alloVector(1))
    !ERROR: Rank of dummy argument is 1, but actual argument has rank 2
    call allocatable1(alloMatrix)
    !ERROR: Rank of dummy argument is 1, but actual argument has rank 0
    !ERROR: ALLOCATABLE dummy argument 'a=' must be associated with an ALLOCATABLE actual argument
    call allocatable1(alloMatrix(1,1))

    !ERROR: Rank of dummy argument is 2, but actual argument has rank 0
    !ERROR: ALLOCATABLE dummy argument 'a=' must be associated with an ALLOCATABLE actual argument
    call allocatable2(scalar0)
    !ERROR: Rank of dummy argument is 2, but actual argument has rank 0
    call allocatable2(alloScalar)
    !ERROR: Rank of dummy argument is 2, but actual argument has rank 1
    call allocatable2(alloVector)
    !ERROR: Rank of dummy argument is 2, but actual argument has rank 0
    !ERROR: ALLOCATABLE dummy argument 'a=' must be associated with an ALLOCATABLE actual argument
    call allocatable2(alloVector(1))
    call allocatable2(alloMatrix)
    !ERROR: Rank of dummy argument is 2, but actual argument has rank 0
    !ERROR: ALLOCATABLE dummy argument 'a=' must be associated with an ALLOCATABLE actual argument
    call allocatable2(alloMatrix(1,1))

    call pointer0(scalar0)
    !ERROR: Rank of dummy argument is 0, but actual argument has rank 1
    !ERROR: Pointer has rank 0 but target has rank 1
    call pointer0(vector1)
    call pointer0(vector1(1))
    !ERROR: Rank of dummy argument is 0, but actual argument has rank 2
    !ERROR: Pointer has rank 0 but target has rank 2
    call pointer0(matrix11)
    call pointer0(matrix11(1,1))

    !ERROR: Rank of dummy argument is 1, but actual argument has rank 0
    !ERROR: Pointer has rank 1 but target has rank 0
    call pointer1(scalar0)
    call pointer1(vector1)
    !ERROR: Rank of dummy argument is 1, but actual argument has rank 0
    !ERROR: Pointer has rank 1 but target has rank 0
    call pointer1(vector1(1))
    !ERROR: Rank of dummy argument is 1, but actual argument has rank 2
    !ERROR: Pointer has rank 1 but target has rank 2
    call pointer1(matrix11)
    !ERROR: Rank of dummy argument is 1, but actual argument has rank 0
    !ERROR: Pointer has rank 1 but target has rank 0
    call pointer1(matrix11(1,1))

    !ERROR: Rank of dummy argument is 2, but actual argument has rank 0
    !ERROR: Pointer has rank 2 but target has rank 0
    call pointer2(scalar0)
    !ERROR: Rank of dummy argument is 2, but actual argument has rank 1
    !ERROR: Pointer has rank 2 but target has rank 1
    call pointer2(vector1)
    !ERROR: Rank of dummy argument is 2, but actual argument has rank 0
    !ERROR: Pointer has rank 2 but target has rank 0
    call pointer2(vector1(1))
    call pointer2(matrix11)
    !ERROR: Rank of dummy argument is 2, but actual argument has rank 0
    !ERROR: Pointer has rank 2 but target has rank 0
    call pointer2(matrix11(1,1))

    !ERROR: Actual argument associated with coarray dummy argument 'a=' must be a coarray
    call coarray0(scalar0)
    !ERROR: Rank of dummy argument is 0, but actual argument has rank 1
    !ERROR: Actual argument associated with coarray dummy argument 'a=' must be a coarray
    call coarray0(vector1)
    !ERROR: Actual argument associated with coarray dummy argument 'a=' must be a coarray
    call coarray0(vector1(1))
    !ERROR: Rank of dummy argument is 0, but actual argument has rank 2
    !ERROR: Actual argument associated with coarray dummy argument 'a=' must be a coarray
    call coarray0(matrix11)
    !ERROR: Actual argument associated with coarray dummy argument 'a=' must be a coarray
    call coarray0(matrix11(1,1))

    !WARNING: Actual argument variable length '1' is less than expected length '2'
    call scalar(scalar0(1:1))
    !WARNING: Actual argument expression length '1' is less than expected length '2'
    call scalar('a')
  end
end

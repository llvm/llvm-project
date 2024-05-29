! RUN: %python %S/test_errors.py %s %flang_fc1
! Tests for F'2023 C1132:
! A variable-name that appears in a REDUCE locality-spec shall be of intrinsic
! type suitable for the intrinsic operation or function specified by its
! reduce-operation.

subroutine s1(n)
! This is OK
  integer :: i1, i2, i3, i4, i5, i6, i7, n
  real(8) :: r1, r2, r3, r4
  complex :: c1, c2
  logical :: l1, l2, l3(n,n), l4(n)
  do concurrent(i=1:5) &
       & reduce(+:i1,r1,c1) reduce(*:i2,r2,c2) reduce(iand:i3) reduce(ieor:i4) &
       & reduce(ior:i5) reduce(max:i6,r3) reduce(min:i7,r4) reduce(.and.:l1) &
       & reduce(.or.:l2) reduce(.eqv.:l3) reduce(.neqv.:l4)
  end do
end subroutine s1

subroutine s2()
! Cannot apply logical operations to integer variables
  integer :: i1, i2, i3, i4
!ERROR: Reduction variable 'i1' ('INTEGER(4)') does not have a suitable type ('LOGICAL').
!ERROR: Reduction variable 'i2' ('INTEGER(4)') does not have a suitable type ('LOGICAL').
!ERROR: Reduction variable 'i3' ('INTEGER(4)') does not have a suitable type ('LOGICAL').
!ERROR: Reduction variable 'i4' ('INTEGER(4)') does not have a suitable type ('LOGICAL').
  do concurrent(i=1:5) &
       & reduce(.and.:i1) reduce(.or.:i2) reduce(.eqv.:i3) reduce(.neqv.:i4)
  end do
end subroutine s2

subroutine s3()
! Cannot apply integer/logical operations to real variables
  real :: r1, r2, r3, r4
!ERROR: Reduction variable 'r1' ('REAL(4)') does not have a suitable type ('INTEGER').
!ERROR: Reduction variable 'r2' ('REAL(4)') does not have a suitable type ('INTEGER').
!ERROR: Reduction variable 'r3' ('REAL(4)') does not have a suitable type ('INTEGER').
!ERROR: Reduction variable 'r4' ('REAL(4)') does not have a suitable type ('LOGICAL').
!ERROR: Reduction variable 'r5' ('REAL(4)') does not have a suitable type ('LOGICAL').
!ERROR: Reduction variable 'r6' ('REAL(4)') does not have a suitable type ('LOGICAL').
!ERROR: Reduction variable 'r7' ('REAL(4)') does not have a suitable type ('LOGICAL').
  do concurrent(i=1:5) &
       & reduce(iand:r1) reduce(ieor:r2) reduce(ior:r3) reduce(.and.:r4) &
       & reduce(.or.:r5) reduce(.eqv.:r6) reduce(.neqv.:r7)
  end do
end subroutine s3

subroutine s4()
! Cannot apply integer/logical operations to complex variables
  complex :: c1, c2, c3, c4, c5, c6, c7, c8, c9
!ERROR: Reduction variable 'c1' ('COMPLEX(4)') does not have a suitable type ('INTEGER').
!ERROR: Reduction variable 'c2' ('COMPLEX(4)') does not have a suitable type ('INTEGER').
!ERROR: Reduction variable 'c3' ('COMPLEX(4)') does not have a suitable type ('INTEGER').
!ERROR: Reduction variable 'c4' ('COMPLEX(4)') does not have a suitable type ('INTEGER', or 'REAL').
!ERROR: Reduction variable 'c5' ('COMPLEX(4)') does not have a suitable type ('INTEGER', or 'REAL').
!ERROR: Reduction variable 'c6' ('COMPLEX(4)') does not have a suitable type ('LOGICAL').
!ERROR: Reduction variable 'c7' ('COMPLEX(4)') does not have a suitable type ('LOGICAL').
!ERROR: Reduction variable 'c8' ('COMPLEX(4)') does not have a suitable type ('LOGICAL').
!ERROR: Reduction variable 'c9' ('COMPLEX(4)') does not have a suitable type ('LOGICAL').
  do concurrent(i=1:5) &
       & reduce(iand:c1) reduce(ieor:c2) reduce(ior:c3) reduce(max:c4) &
       & reduce(min:c5) reduce(.and.:c6) reduce(.or.:c7) reduce(.eqv.:c8) &
       & reduce(.neqv.:c9)
  end do
end subroutine s4

subroutine s5()
! Cannot apply integer operations to logical variables
  logical :: l1, l2, l3, l4, l5, l6, l7
!ERROR: Reduction variable 'l1' ('LOGICAL(4)') does not have a suitable type ('COMPLEX', 'INTEGER', or 'REAL').
!ERROR: Reduction variable 'l2' ('LOGICAL(4)') does not have a suitable type ('COMPLEX', 'INTEGER', or 'REAL').
!ERROR: Reduction variable 'l3' ('LOGICAL(4)') does not have a suitable type ('INTEGER').
!ERROR: Reduction variable 'l4' ('LOGICAL(4)') does not have a suitable type ('INTEGER').
!ERROR: Reduction variable 'l5' ('LOGICAL(4)') does not have a suitable type ('INTEGER').
!ERROR: Reduction variable 'l6' ('LOGICAL(4)') does not have a suitable type ('INTEGER', or 'REAL').
!ERROR: Reduction variable 'l7' ('LOGICAL(4)') does not have a suitable type ('INTEGER', or 'REAL').
  do concurrent(i=1:5) &
       & reduce(+:l1) reduce(*:l2) reduce(iand:l3) reduce(ieor:l4) &
       & reduce(ior:l5) reduce(max:l6) reduce(min:l7)
  end do
end subroutine s5

subroutine s6()
! Cannot reduce a character
  character ch
!ERROR: Reduction variable 'ch' ('CHARACTER(1_8,1)') does not have a suitable type ('COMPLEX', 'INTEGER', or 'REAL').
  do concurrent(i=1:5) reduce(+:ch)
  end do
end subroutine s6

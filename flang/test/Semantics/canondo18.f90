! RUN: %S/test_any.sh %s %flang %t
! Error test -- DO loop uses obsolete loop termination statement
! See R1131 and C1133

! By default, this is not an error and label do are rewritten to non-label do.
! A warning is generated with -Mstandard

! EXEC: ${F18} -funparse-with-symbols -Mstandard %s 2>&1 | ${FileCheck} %s

! CHECK: end do

! The following CHECK-NOT actively uses the fact that the leading zero of labels
! would be removed in the unparse but not the line linked to warnings. We do
! not want to see label do in the unparse only.
! CHECK-NOT: do [1-9]

! CHECK: A DO loop should terminate with an END DO or CONTINUE

subroutine foo10()
  real :: a(10, 10), b(10, 10) = 1.0
  do 01 k=1,4
    where (a<k)
      a = a + b
      b = a - b
    elsewhere
      a = a*2
01  end where
end subroutine

! Error test -- DO loop uses obsolete loop termination statement (warning)
! See R1131 and C1133

! By default, this is not an error and label do are rewritten to non-label do.
! A warning is generated with -Mstandard


! RUN: %f18 -funparse-with-symbols -Mstandard %s 2>%t.stderr | FileCheck %s

! CHECK: end do

! The following CHECK-NOT actively uses the fact that the leading zero of labels
! would be removed in the unparse but not the line linked to warnings. We do
! not want to see label do in the unparse only.
! CHECK-NOT: do [1-9]

! RUN: FileCheck --check-prefix=ERR --input-file=%t.stderr %s
! ERR: A DO loop should terminate with an END DO or CONTINUE

subroutine foo3()
  real :: a(10, 10), b(10, 10) = 1.0
  do 01 k=1,4
    associate (x=>a(k+1, 2*k), y=>b(k, 2*k-1))
      x = 4*x*x + x*y -2*y
01  end associate
end subroutine

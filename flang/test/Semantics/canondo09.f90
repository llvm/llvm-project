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

subroutine foo0()
  do  01 j=1,2
    if (.true.) then
01  end if
end subroutine

! Test -S (SystemZ)

! REQUIRES: systemz-registered-target

! RUN: %flang_fc1 -S -triple s390x-unknown-linux-gnu %s -o - | FileCheck %s
! RUN: %flang -S -target s390x-unknown-linux-gnu %s -o - | FileCheck %s

! CHECK-LABEL: _QQmain:
! CHECK: br %r14

end program

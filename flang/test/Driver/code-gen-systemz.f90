! Test -emit-obj for SystemZ (s390x)

! REQUIRES: systemz-registered-target

! RUN: %flang_fc1 -triple s390x-unknown-linux-gnu -emit-obj %s -o - | \
! RUN: llvm-readobj -h - | FileCheck %s

! CHECK: Arch: s390x
end program

! Test -emit-obj (X86)

! REQUIRES: aarch64-registered-target

! RUN: %flang_fc1 -triple aarch64-unknown-linux-gnu -emit-obj %s -o - | \
! RUN: llvm-readobj -h - | FileCheck %s

! CHECK: Arch: aarch64
end program

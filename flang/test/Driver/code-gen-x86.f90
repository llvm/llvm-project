! Test -emit-obj (X86)

! REQUIRES: x86-registered-target

! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -emit-obj %s -o - | \
! RUN: llvm-readobj -h - | FileCheck %s

! CHECK: Arch: x86_64
end program

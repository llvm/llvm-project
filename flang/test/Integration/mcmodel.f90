!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! REQUIRES:  aarch64-registered-target && x86-registered-target

! RUN: %flang_fc1 -triple aarch64 -emit-llvm -mcmodel=tiny %s -o - | FileCheck %s -check-prefix=CHECK-TINY
! RUN: %flang_fc1 -emit-llvm -mcmodel=small %s -o - | FileCheck %s -check-prefix=CHECK-SMALL
! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -emit-llvm -mcmodel=kernel %s -o - | FileCheck %s -check-prefix=CHECK-KERNEL
! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -emit-llvm -mcmodel=medium %s -o - | FileCheck %s -check-prefix=CHECK-MEDIUM
! RUN: %flang_fc1 -emit-llvm -mcmodel=large %s -o - | FileCheck %s -check-prefix=CHECK-LARGE

! CHECK-TINY: !llvm.module.flags = !{{{.*}}}
! CHECK-TINY: !{{[0-9]+}} = !{i32 1, !"Code Model", i32 0}
! CHECK-SMALL: !llvm.module.flags = !{{{.*}}}
! CHECK-SMALL: !{{[0-9]+}} = !{i32 1, !"Code Model", i32 1}
! CHECK-KERNEL: !llvm.module.flags = !{{{.*}}}
! CHECK-KERNEL: !{{[0-9]+}} = !{i32 1, !"Code Model", i32 2}
! CHECK-MEDIUM: !llvm.module.flags = !{{{.*}}}
! CHECK-MEDIUM: !{{[0-9]+}} = !{i32 1, !"Code Model", i32 3}
! CHECK-LARGE: !llvm.module.flags = !{{{.*}}}
! CHECK-LARGE: !{{[0-9]+}} = !{i32 1, !"Code Model", i32 4}

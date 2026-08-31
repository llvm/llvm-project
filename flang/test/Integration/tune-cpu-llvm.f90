!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! RUN: %if x86-registered-target %{ %flang -target x86_64-linux-gnu -mtune=pentium4 -S -emit-llvm %s -o - | FileCheck %s --check-prefixes=ALL,CHECK-X86 %}
! RUN: %if aarch64-registered-target %{ %flang -target aarch64-linux-gnu -mtune=neoverse-n1 -S -emit-llvm %s -o - | FileCheck %s --check-prefixes=ALL,CHECK-ARM %}

!ALL: attributes #{{[0-9]+}} = {
!CHECK-X86-SAME: "tune-cpu"="pentium4"
!CHECK-ARM-SAME: "tune-cpu"="neoverse-n1"
subroutine a
end subroutine a

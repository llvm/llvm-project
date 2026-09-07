! REQUIRES: x86-registered-target

! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -emit-obj -debug-info-kind=standalone %s -o %t.o
! RUN: llvm-readelf -r %t.o | FileCheck %s
! RUN: llvm-readelf --symbols %t.o | FileCheck %s --check-prefix=NO_UND

! Test that the object file has no undefined symbol from iso_fortran_env, which
! means the debug information leaves no unresolved relocation behind.

program p
  use iso_fortran_env
  implicit none
  print *, 'hello'
end program p

! CHECK: .rela.debug_info

! NO_UND-NOT: UND{{.*}}_QMiso_fortran_env

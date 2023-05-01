! Test that a fortran main program can be linked to an executable
! by flang.
!
! For now, this test only covers the Gnu toolchain on linux.

!REQUIRES: x86-registered-target || aarch64-registered-target || riscv64-registered-target
!REQUIRES: system-linux

! RUN: %flang_fc1 -emit-obj %s -o %t.o
! RUN: %flang -target x86_64-unknown-linux-gnu %t.o -o %t.out -flang-experimental-exec
! RUN: llvm-objdump --syms %t.out | FileCheck %s

! Test that it also works if the program is bundled in an archive.

! RUN: llvm-ar -r %t.a %t.o
! RUN: %flang -target x86_64-unknown-linux-gnu %t.a -o %ta.out -flang-experimental-exec
! RUN: llvm-objdump --syms %ta.out | FileCheck %s

end program

! CHECK-DAG: F .text {{[a-f0-9]+}} main
! CHECK-DAG: F .text {{[a-f0-9]+}} _QQmain
! CHECK-DAG: _FortranAProgramStart

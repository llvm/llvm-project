! REQUIRES: x86-registered-target, zlib

! Test that --compress-debug-sections=zlib compresses debug sections in the
! output object file.
! RUN: %flang_fc1 -debug-info-kind=standalone -triple x86_64-unknown-linux \
! RUN:   --compress-debug-sections=zlib -emit-obj -o %t.o %s
! RUN: llvm-readobj -S %t.o | FileCheck --check-prefix=ZLIB %s

! ZLIB: Name: .debug_info
! ZLIB-NOT: Section
! ZLIB: SHF_COMPRESSED

! Test that --compress-debug-sections=none does not compress debug sections.
! RUN: %flang_fc1 -debug-info-kind=standalone -triple x86_64-unknown-linux \
! RUN:   --compress-debug-sections=none -emit-obj -o %t_none.o %s
! RUN: llvm-readobj -S %t_none.o | FileCheck --check-prefix=NONE %s

! NONE: Name: .debug_info
! NONE-NOT: Section
! NONE: Flags [
! NONE-NOT: SHF_COMPRESSED

program test
  implicit none
  integer :: x(1000)
  x = 1
  print *, x(1000)
end program test

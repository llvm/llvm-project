! Test predefined macro for 64 bit X86 architecture

! REQUIRES: x86-registered-target

! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -cpp -E %s | FileCheck %s

! CHECK: integer :: var1 = 1
! CHECK: integer :: var2 = 1

#if __x86_64__
  integer :: var1 = __x86_64__
#endif
#if __x86_64__
  integer :: var2 = __x86_64
#endif
end program

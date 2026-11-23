! Test predefined macro for AArch64

! REQUIRES: aarch64-registered-target

! RUN: %flang_fc1 -triple aarch64-unknown-linux-gnu -cpp -E %s | FileCheck %s

! CHECK: integer :: var1 = 1
! CHECK: integer :: var2 = 1

#if __aarch64__
  integer :: var1 = __aarch64__
#endif
#if __aarch64
  integer :: var2 = __aarch64
#endif
end program

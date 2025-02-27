! Test predefined macro for PowerPC architecture

! RUN: %flang_fc1 -cpp -E %s | FileCheck %s
! REQUIRES: target=powerpc{{.*}}

! CHECK: integer :: var1 = 1

#if __powerpc__
  integer :: var1 = __powerpc__
#endif
end program

! Test predefined macro for PowerPC architecture

! RUN: %flang_fc1 -triple ppc64le-unknown-linux -cpp -E %s | FileCheck %s
! REQUIRES: target=powerpc{{.*}}

! CHECK: integer :: var1 = 1
! CHECK: integer :: var2 = 1

#if defined(__linux__) && defined(__powerpc__)
  integer :: var1 = __powerpc__
  integer :: var2 = __linux__
#endif
end program

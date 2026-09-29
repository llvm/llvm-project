! Test predefined macro for PowerPC architecture

! RUN: %flang_fc1 -triple ppc64le-unknown-linux -cpp -E %s | FileCheck %s -check-prefix=CHECK-LINUX
! RUN: %flang_fc1 -triple powerpc-unknown-aix -cpp -E %s | FileCheck %s -check-prefix=CHECK-AIX32
! RUN: %flang_fc1 -triple powerpc64-unknown-aix -cpp -E %s | FileCheck %s -check-prefix=CHECK-AIX64
! REQUIRES: target=powerpc{{.*}}

! CHECK-LINUX: integer :: var1 = 1
! CHECK-LINUX: integer :: var2 = 1
! CHECK-AIX32: integer :: var1 = 1
! CHECK-AIX32: integer :: var2 = 1
! CHECK-AIX32: integer :: var3 = __64BIT__
! CHECK-AIX64: integer :: var1 = 1
! CHECK-AIX64: integer :: var2 = 1
! CHECK-AIX64: integer :: var3 = 1

#if defined(__linux__) && defined(__powerpc__)
  integer :: var1 = __powerpc__
  integer :: var2 = __linux__
#elif defined(_AIX) && defined(__powerpc__)
  integer :: var1 = __powerpc__
  integer :: var2 = _AIX
  integer :: var3 = __64BIT__
#endif
end program

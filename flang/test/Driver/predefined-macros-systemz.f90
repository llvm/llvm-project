! Test predefined macros for SystemZ architecture
! REQUIRES: systemz-registered-target

! RUN: %flang_fc1 -triple s390x-unknown-linux-gnu -cpp -E %s | FileCheck %s

! CHECK: integer :: var1 = 1
! CHECK: integer :: var2 = 1
! CHECK: integer :: var3 = 1
! CHECK: integer :: var4 = 1

#if __s390__
  integer :: var1 = __s390__
#endif
#if __s390x__
  integer :: var2 = __s390x__
#endif
#if __s390x
  integer :: var3 = __s390x
#endif
#if __zarch__
  integer :: var4 = __zarch__
#endif
end program

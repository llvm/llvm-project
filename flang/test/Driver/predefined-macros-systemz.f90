! Test predefined macros for SystemZ architecture
! REQUIRES: systemz-registered-target

! RUN: %flang_fc1 -triple s390x-unknown-linux-gnu -cpp -E %s | FileCheck %s

! CHECK: integer :: var1 = 1
! CHECK: integer :: var2 = 1

#if __s390x__
  integer :: var1 = __s390x__
#endif
#if __s390x
  integer :: var2 = __s390x
#endif
end program

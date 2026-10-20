! RUN: %flang -E %s 2>&1 | FileCheck %s
#ifndef XYZ42 
      PARAMETER(A=2)
#else Z
      PARAMETER(A=3)
#endif
! Ensure that "PARAMETER(A" is printed only once
! CHECK: PARAMETER(A
! CHECK-NOT: PARAMETER(A
      end


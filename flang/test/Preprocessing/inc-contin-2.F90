! RUN: %flang_fc1 -E %s 2>&1 | FileCheck %s
! CHECK: print *, 3.14159
! CHECK: print *, 3. 14159
      program main
#include "inc-contin-2a.h"
     &14159
#include "inc-contin-2b.h"
     &14159
      end program main

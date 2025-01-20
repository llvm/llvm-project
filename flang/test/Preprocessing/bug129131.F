! RUN: %flang -fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
! CHECK: PRINT *, 2_4
#define a ,3
      print *, mod(5 a)
      end

! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK: call foo (  3.14159)
! CHECK: subroutine foo(test)
      call foo (
#include "ff-args.h"
      end
#define TEST
      subroutine foo(
#ifdef TEST
     +test)
#else
     +)
#endif
      end

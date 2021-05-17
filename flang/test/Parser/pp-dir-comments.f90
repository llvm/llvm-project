! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s

#define pmk
#ifdef pmk // comment
! CHECK: t1
real t1
#endif // comment
#undef pmk ! comment
#ifndef pmk ! comment
! CHECK: t2
real t2
#endif // comment
#if 0 /* C comment */ + 0
! CHECK-NOT: misinterpreted
# error misinterpreted #if
#else // comment
! CHECK: END PROGRAM
end
#endif ! comment

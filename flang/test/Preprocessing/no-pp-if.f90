!RUN: %flang -fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
!CHECK-NOT: ERROR STOP
!CHECK: CONTINUE
#if defined UNDEFINED
error stop
#endif
#if !defined UNDEFINED
continue
#endif
end

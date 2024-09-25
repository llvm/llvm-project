! RUN: %flang_fc1 -E %s 2>&1 | FileCheck %s
! CHECK: subroutine test(ARG1, FA, FB, ARG2)
! CHECK: end

subroutine test( &
ARG1, &
! test
#ifndef SWAP
#define ARG1 FA
#define ARG2 FB
#else
#define ARG1 FB
#define ARG2 FA
#endif
ARG1, ARG2, &
! test
#undef ARG1
#undef ARG2
&ARG2)
! comment
end

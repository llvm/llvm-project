!RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
!CHECK: warning: REAL(4) to INTEGER(2) conversion overflowed
print *, int2(4.e9), int8(4.e9)
!CHECK: error: 'int2' is not an unrestricted specific intrinsic procedure
!CHECK: error: 'int8' is not an unrestricted specific intrinsic procedure
call foo(int2,int8)
end

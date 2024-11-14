! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
!CHECK: warning: REAL(4) to INTEGER(2) conversion overflowed
!CHECK: PRINT *, 32767_2, 4000000000_8
print *, int2(4.e9), int8(4.e9)
end

! RUN: %flang_fc1 -fdebug-unparse %s -Dj=1 2>&1 | FileCheck %s
! Ensure that macro definitions don't affect INCLUDE lines (unlike #include)
#define sin cos
!CHECK: PRINT *, 0._4, j
include "include-file.h"
end

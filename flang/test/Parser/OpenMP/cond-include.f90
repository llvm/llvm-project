! RUN: %flang_fc1 -fdebug-unparse -fopenmp %s | FileCheck %s
!CHECK: STOP "pass"
!$ include "cond-include.inc"
end

!RUN: %flang_fc1 -fdebug-unparse -fopenmp %s 2>&1 | FileCheck %s
!CHECK: !$OMP DECLARE TARGET
#define OMP_DECLARE_TARGET $OMP declare target
subroutine s
  !OMP_DECLARE_TARGET
end


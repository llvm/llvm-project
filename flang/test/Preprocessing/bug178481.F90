!RUN: %flang_fc1 -fdebug-unparse -fopenmp %s 2>&1 | FileCheck %s
#define OMP_DECLARE_TARGET $OMP declare target
#define OMP_BANG $OMP
subroutine s
  !CHECK: !$OMP DECLARE TARGET
  !OMP_DECLARE_TARGET
  !CHECK: !$OMP DECLARE TARGET
  !OMP_BANG declare &
  !OMP_BANG target
end

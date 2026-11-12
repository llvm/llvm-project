!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! This should pass semantic checks.

!CHECK: func.func

subroutine f
  10 continue
end

subroutine g
  !$omp parallel
  goto 10
  10 continue
  !$omp end parallel
end

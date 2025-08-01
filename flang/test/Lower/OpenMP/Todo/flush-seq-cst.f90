! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -o - %s -fopenmp-version=51 2>&1 | FileCheck %s

! CHECK: not yet implemented: Unhandled clause SEQ_CST in FLUSH construct
program flush_seq_cst
  !$omp flush seq_cst
end program
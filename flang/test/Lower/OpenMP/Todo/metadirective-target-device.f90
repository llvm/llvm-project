! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: target_device={device_num()} selector in METADIRECTIVE

subroutine test_target_device_num()
  continue
  !$omp metadirective &
  !$omp & when(target_device={device_num(0), kind(gpu)}: barrier) &
  !$omp & default(nothing)
end subroutine

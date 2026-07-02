! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: target_device selector in DECLARE VARIANT

subroutine test_target_device
  call base()
end subroutine

subroutine base
  !$omp declare variant (base:vsub) match (target_device={kind(host)})
contains
  subroutine vsub
  end subroutine
end subroutine

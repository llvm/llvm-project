! RUN: %flang_fc1 -fdebug-unparse -fopenacc %s 2>&1 | FileCheck %s
! CHECK: !$ACC DECLARE COPYIN(var)
#define ACCDECLARE(var) integer :: var; \
  !$acc declare copyin(var)
program main
  ACCDECLARE(var)
end program

! A breakpoint on a scalar broadcast must stop once at -O0, even though
! the assignment is expanded into a loop. Require exactly one statement row.
! REQUIRES: x86-registered-target

! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -emit-obj -debug-info-kind=line-tables-only -O0 %s -o %t.o && llvm-dwarfdump --debug-line %t.o | FileCheck %s

subroutine host_scalar_broadcast(arr)
  integer :: arr(4)
  ! CHECK: 0x{{[0-9a-f]+}} [[#@LINE+2]] {{.*}} is_stmt
  ! CHECK-NOT: 0x{{[0-9a-f]+}} [[#@LINE+1]] {{.*}} is_stmt
  arr = 11
end subroutine

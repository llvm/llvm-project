! RUN: %flang -fopenmp -S -emit-llvm %s -o - | FileCheck %s
subroutine sub1
  implicit none
  integer :: x

  write (*, *) "HELLO1"
  entry sub2
  x = 0
  write (*, *) "HELLO2"
!$OMP CRITICAL
  x = x + 1
!$OMP END CRITICAL
end subroutine
! CHECK: call i32 @__kmpc_global_thread_num
! CHECK: call i32 @__kmpc_global_thread_num


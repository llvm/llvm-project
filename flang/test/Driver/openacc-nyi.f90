! Verify that OpenACC NYI diagnostics are delayed until code generation and
! use Flang's TODO facility for every source-level directive.

! RUN: split-file %s %t
! RUN: %flang_fc1 -fopenacc -emit-hlfir %t/parallel.f90
! RUN: %flang_fc1 -fopenacc -emit-hlfir %t/kernels.f90
! RUN: %flang_fc1 -fopenacc -emit-hlfir %t/serial.f90
! RUN: %flang_fc1 -fopenacc -emit-hlfir %t/parallel-loop.f90
! RUN: %flang_fc1 -fopenacc -emit-hlfir %t/kernels-loop.f90
! RUN: %flang_fc1 -fopenacc -emit-hlfir %t/serial-loop.f90
! RUN: %flang_fc1 -fopenacc -emit-hlfir %t/data.f90
! RUN: %flang_fc1 -fopenacc -emit-hlfir %t/loop.f90
! RUN: %flang_fc1 -fopenacc -emit-hlfir %t/enter-data.f90
! RUN: %flang_fc1 -fopenacc -emit-hlfir %t/exit-data.f90
! RUN: %flang_fc1 -fopenacc -emit-hlfir %t/host-data.f90
! RUN: %flang_fc1 -fopenacc -emit-hlfir %t/init.f90
! RUN: %flang_fc1 -fopenacc -emit-hlfir %t/shutdown.f90
! RUN: %flang_fc1 -fopenacc -emit-hlfir %t/update.f90
! RUN: %flang_fc1 -fopenacc -emit-hlfir %t/set.f90
! RUN: %flang_fc1 -fopenacc -emit-hlfir %t/wait.f90
! RUN: %flang_fc1 -fopenacc -emit-hlfir %t/atomic.f90
! RUN: %flang_fc1 -fopenacc -emit-hlfir %t/routine.f90
! RUN: %flang_fc1 -fopenacc -emit-hlfir %t/declare.f90
! RUN: %flang_fc1 -fopenacc -emit-hlfir %t/declare-module-global.f90
! RUN: %flang_fc1 -fopenacc -emit-hlfir %t/cache.f90
! RUN: %not_todo_cmd %flang -fopenacc -S -emit-llvm %t/parallel.f90 2>&1 | FileCheck %t/parallel.f90
! RUN: %not_todo_cmd %flang -fopenacc -S -emit-llvm %t/kernels.f90 2>&1 | FileCheck %t/kernels.f90
! RUN: %not_todo_cmd %flang -fopenacc -S -emit-llvm %t/serial.f90 2>&1 | FileCheck %t/serial.f90
! RUN: %not_todo_cmd %flang -fopenacc -S -emit-llvm %t/parallel-loop.f90 2>&1 | FileCheck %t/parallel-loop.f90
! RUN: %not_todo_cmd %flang -fopenacc -S -emit-llvm %t/kernels-loop.f90 2>&1 | FileCheck %t/kernels-loop.f90
! RUN: %not_todo_cmd %flang -fopenacc -S -emit-llvm %t/serial-loop.f90 2>&1 | FileCheck %t/serial-loop.f90
! RUN: %not_todo_cmd %flang -fopenacc -S -emit-llvm %t/data.f90 2>&1 | FileCheck %t/data.f90
! RUN: %not_todo_cmd %flang -fopenacc -S -emit-llvm %t/loop.f90 2>&1 | FileCheck %t/loop.f90
! RUN: %not_todo_cmd %flang -fopenacc -S -emit-llvm %t/enter-data.f90 2>&1 | FileCheck %t/enter-data.f90
! RUN: %not_todo_cmd %flang -fopenacc -S -emit-llvm %t/exit-data.f90 2>&1 | FileCheck %t/exit-data.f90
! RUN: %not_todo_cmd %flang -fopenacc -S -emit-llvm %t/host-data.f90 2>&1 | FileCheck %t/host-data.f90
! RUN: %not_todo_cmd %flang -fopenacc -S -emit-llvm %t/init.f90 2>&1 | FileCheck %t/init.f90
! RUN: %not_todo_cmd %flang -fopenacc -S -emit-llvm %t/shutdown.f90 2>&1 | FileCheck %t/shutdown.f90
! RUN: %not_todo_cmd %flang -fopenacc -S -emit-llvm %t/update.f90 2>&1 | FileCheck %t/update.f90
! RUN: %not_todo_cmd %flang -fopenacc -S -emit-llvm %t/set.f90 2>&1 | FileCheck %t/set.f90
! RUN: %not_todo_cmd %flang -fopenacc -S -emit-llvm %t/wait.f90 2>&1 | FileCheck %t/wait.f90
! RUN: %not_todo_cmd %flang -fopenacc -S -emit-llvm %t/atomic.f90 2>&1 | FileCheck %t/atomic.f90
! RUN: %not_todo_cmd %flang -fopenacc -S -emit-llvm %t/routine.f90 2>&1 | FileCheck %t/routine.f90
! RUN: %not_todo_cmd %flang -fopenacc -S -emit-llvm %t/declare.f90 2>&1 | FileCheck %t/declare.f90
! RUN: %not_todo_cmd %flang -fopenacc -S -emit-llvm %t/declare-module-global.f90 2>&1 | FileCheck %t/declare-module-global.f90
! RUN: %not_todo_cmd %flang -fopenacc -S -emit-llvm %t/cache.f90 2>&1 | FileCheck %t/cache.f90

!--- parallel.f90
! CHECK: not yet implemented: OpenACC parallel directive
subroutine test
  !$acc parallel
  !$acc end parallel
end

!--- kernels.f90
! CHECK: not yet implemented: OpenACC kernels directive
subroutine test
  !$acc kernels
  !$acc end kernels
end

!--- serial.f90
! CHECK: not yet implemented: OpenACC serial directive
subroutine test
  !$acc serial
  !$acc end serial
end

!--- parallel-loop.f90
! CHECK: not yet implemented: OpenACC parallel loop directive
subroutine test
  integer i
  !$acc parallel loop
  do i = 1, 10
  end do
end

!--- kernels-loop.f90
! CHECK: not yet implemented: OpenACC kernels loop directive
subroutine test
  integer i
  !$acc kernels loop
  do i = 1, 10
  end do
end

!--- serial-loop.f90
! CHECK: not yet implemented: OpenACC serial loop directive
subroutine test
  integer i
  !$acc serial loop
  do i = 1, 10
  end do
end

!--- data.f90
! CHECK: not yet implemented: OpenACC data directive
subroutine test(a)
  real a
  !$acc data copy(a)
  !$acc end data
end

!--- loop.f90
! CHECK: not yet implemented: OpenACC loop directive
subroutine test
  integer i
  !$acc loop
  do i = 1, 10
  end do
end

!--- enter-data.f90
! CHECK: not yet implemented: OpenACC enter data directive
subroutine test(a)
  real a
  !$acc enter data copyin(a)
end

!--- exit-data.f90
! CHECK: not yet implemented: OpenACC exit data directive
subroutine test(a)
  real a
  !$acc exit data delete(a)
end

!--- host-data.f90
! CHECK: not yet implemented: OpenACC host_data directive
subroutine test(a)
  real a
  !$acc host_data use_device(a)
  !$acc end host_data
end

!--- init.f90
! CHECK: not yet implemented: OpenACC init directive
subroutine test
  !$acc init
end

!--- shutdown.f90
! CHECK: not yet implemented: OpenACC shutdown directive
subroutine test
  !$acc shutdown
end

!--- update.f90
! CHECK: not yet implemented: OpenACC update directive
subroutine test(a)
  real a
  !$acc update device(a)
end

!--- set.f90
! CHECK: not yet implemented: OpenACC set directive
subroutine test
  !$acc set device_num(0)
end

!--- wait.f90
! CHECK: not yet implemented: OpenACC wait directive
subroutine test
  !$acc wait
end

!--- atomic.f90
! CHECK: not yet implemented: OpenACC atomic directive
subroutine test(a)
  integer a
  !$acc atomic update
  a = a + 1
end

!--- routine.f90
! CHECK: not yet implemented: OpenACC routine directive
subroutine test
  !$acc routine seq
end

!--- declare.f90
! CHECK: not yet implemented: OpenACC declare directive
subroutine test
  real a
  !$acc declare create(a)
end

!--- declare-module-global.f90
! CHECK: not yet implemented: OpenACC declare directive
module test
  real :: a
  !$acc declare create(a)
end module

!--- cache.f90
! CHECK: not yet implemented: OpenACC cache directive
subroutine test(a)
  real a(10)
  !$acc cache(a)
end

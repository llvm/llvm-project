! Test metadirective with device={arch(...)} trait selectors.

! RUN: %if aarch64-registered-target %{ %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=50 -triple aarch64-unknown-linux-gnu %s -o - | FileCheck --check-prefix=AARCH64 %s %}
! RUN: %if x86-registered-target %{ %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=50 -triple x86_64-unknown-linux-gnu %s -o - | FileCheck --check-prefix=X86_64 %s %}

! AARCH64-LABEL: func.func @_QPtest_arch_aarch64()
! AARCH64:         omp.barrier
! X86_64-LABEL: func.func @_QPtest_arch_aarch64()
! X86_64-NOT:     omp.barrier
! X86_64:         return
subroutine test_arch_aarch64()
  !$omp metadirective &
  !$omp & when(device={arch(aarch64)}: barrier) &
  !$omp & default(nothing)
end subroutine

! AARCH64-LABEL: func.func @_QPtest_arch_x86_64()
! AARCH64-NOT:     omp.barrier
! AARCH64:         return
! X86_64-LABEL: func.func @_QPtest_arch_x86_64()
! X86_64:         omp.barrier
subroutine test_arch_x86_64()
  !$omp metadirective &
  !$omp & when(device={arch(x86_64)}: barrier) &
  !$omp & default(nothing)
end subroutine

! AARCH64-LABEL: func.func @_QPtest_arch_unknown()
! AARCH64-NOT:     omp.barrier
! AARCH64:         omp.taskwait
! X86_64-LABEL: func.func @_QPtest_arch_unknown()
! X86_64-NOT:     omp.barrier
! X86_64:         omp.taskwait
subroutine test_arch_unknown()
  !$omp metadirective &
  !$omp & when(device={arch("unknown_arch")}: barrier) &
  !$omp & default(taskwait)
end subroutine

! AARCH64-LABEL: func.func @_QPtest_begin_arch_aarch64()
! AARCH64:         omp.parallel
! AARCH64:           omp.terminator
! X86_64-LABEL: func.func @_QPtest_begin_arch_aarch64()
! X86_64-NOT:     omp.parallel
! X86_64:         return
subroutine test_begin_arch_aarch64()
  integer :: x
  x = 0
  !$omp begin metadirective &
  !$omp & when(device={arch(aarch64)}: parallel)
  x = 1
  !$omp end metadirective
end subroutine

! AARCH64-LABEL: func.func @_QPtest_begin_arch_x86_64()
! AARCH64-NOT:     omp.parallel
! AARCH64:         return
! X86_64-LABEL: func.func @_QPtest_begin_arch_x86_64()
! X86_64:         omp.parallel
! X86_64:           omp.terminator
subroutine test_begin_arch_x86_64()
  integer :: x
  x = 0
  !$omp begin metadirective &
  !$omp & when(device={arch(x86_64)}: parallel)
  x = 1
  !$omp end metadirective
end subroutine

! AARCH64-LABEL: func.func @_QPtest_begin_arch_unknown()
! AARCH64-NOT:     omp.parallel
! AARCH64:         return
! X86_64-LABEL: func.func @_QPtest_begin_arch_unknown()
! X86_64-NOT:     omp.parallel
! X86_64:         return
subroutine test_begin_arch_unknown()
  integer :: x
  x = 0
  !$omp begin metadirective &
  !$omp & when(device={arch("unknown_arch")}: parallel)
  x = 1
  !$omp end metadirective
end subroutine

! AARCH64-LABEL: func.func @_QPtest_begin_arch_multi_when()
! AARCH64-NOT:     omp.task
! AARCH64:         omp.parallel
! AARCH64:           omp.terminator
! AARCH64-NOT:     omp.task
! AARCH64:         return
! X86_64-LABEL: func.func @_QPtest_begin_arch_multi_when()
! X86_64-NOT:     omp.parallel
! X86_64:         omp.task
! X86_64:           omp.terminator
! X86_64-NOT:     omp.parallel
! X86_64:         return
subroutine test_begin_arch_multi_when()
  integer :: x
  x = 0
  !$omp begin metadirective &
  !$omp & when(device={arch(aarch64)}: parallel) &
  !$omp & when(device={arch(x86_64)}: task)
  x = 1
  !$omp end metadirective
end subroutine

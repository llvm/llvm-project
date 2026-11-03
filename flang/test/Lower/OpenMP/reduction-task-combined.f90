! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -emit-hlfir -mmlir --enable-delayed-privatization=true -o - %s | FileCheck %s

! Check support for reductions with task modifier on compound constructs

! CHECK-LABEL: func.func @_QPcombined_do
subroutine combined_do(x)
  integer :: x, i
  ! CHECK: omp.parallel {
  ! CHECK: omp.wsloop{{.*}}reduction(mod: task, @add_reduction_i32
  !$omp parallel do reduction(task, +: x)
  do i = 1, 4
    x = x + 1
  end do
  !$omp end parallel do
end subroutine

! CHECK-LABEL: func.func @_QPcombined_sections
subroutine combined_sections(x)
  integer :: x
  ! CHECK: omp.parallel {
  ! CHECK: omp.sections reduction(mod: task, @add_reduction_i32
  !$omp parallel sections reduction(task, +: x)
  !$omp section
  x = x + 1
  !$omp end parallel sections
end subroutine

! CHECK-LABEL: func.func @_QPcombined_target
subroutine combined_target(x)
  integer :: x
  ! CHECK: omp.map.info{{.*}}map_clauses(tofrom)
  ! CHECK: omp.target{{.*}}map_entries
  ! CHECK: omp.parallel reduction(mod: task, @add_reduction_i32
  !$omp target parallel reduction(task, +: x)
  x = x + 1
  !$omp end target parallel
end subroutine

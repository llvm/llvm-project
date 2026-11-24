! Test `parallel sections` construct nested inside a `target` region. The
! clause extraction performed while navigating the target region must handle
! the sections construct and preserve the target's `map` clauses.

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPs
subroutine s(x, y)
  integer :: x, y
  ! CHECK: %[[MAP_X:.*]] = omp.map.info {{.*}} map_clauses(tofrom) {{.*}} name("x")
  ! CHECK: %[[MAP_Y:.*]] = omp.map.info {{.*}} map_clauses(tofrom) {{.*}} name("y")
  ! CHECK: omp.target {{.*}} map_entries(%[[MAP_X]] -> {{.*}}, %[[MAP_Y]] -> {{.*}} : {{.*}}) {
  !$omp target map(tofrom: x, y)
  ! CHECK: omp.parallel
  ! CHECK: omp.sections
  !$omp parallel sections
  ! CHECK: omp.section
  !$omp section
    x = 11
  ! CHECK: omp.section
  !$omp section
    y = 22
  !$omp end parallel sections
  !$omp end target
end subroutine

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 %s -o - | FileCheck %s

subroutine map_present_target_data
    integer :: x
!CHECK: %[[MAP:.*]] = omp.map.info {{.*}} map_clauses(present, to) {{.*}} {name = "x"}
!CHECK: omp.target_data map_entries(%[[MAP]] : {{.*}}) {
!$omp target data map(present, to: x)
!$omp end target data
end subroutine

subroutine map_present_update
    integer :: x
!CHECK: %[[MAP:.*]] = omp.map.info {{.*}} map_clauses(present, to) {{.*}} {name = "x"}
!CHECK: omp.target_update map_entries(%[[MAP]] : {{.*}})
!$omp target update to(present: x)
end subroutine

subroutine map_always
    integer :: x
!CHECK: %[[MAP:.*]] = omp.map.info {{.*}} map_clauses(always, tofrom) {{.*}} {name = "x"}
!CHECK: omp.target_data map_entries(%[[MAP]] : {{.*}}) {
!$omp target data map(always, tofrom: x)
!$omp end target data
end subroutine

subroutine map_close
    integer :: x
!CHECK: %[[MAP:.*]] = omp.map.info {{.*}} map_clauses(close, tofrom) {{.*}} {name = "x"}
!CHECK: omp.target_data map_entries(%[[MAP]] : {{.*}}) {
!$omp target data map(close, tofrom: x)
!$omp end target data
end subroutine

subroutine map_ompx_hold
    integer :: x
!CHECK: %[[MAP:.*]] = omp.map.info {{.*}} map_clauses(ompx_hold, tofrom) {{.*}} {name = "x"}
!CHECK: omp.target_data map_entries(%[[MAP]] : {{.*}}) {
!$omp target data map(ompx_hold, tofrom: x)
!$omp end target data
end subroutine

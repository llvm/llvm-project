! RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s

module m
  type :: t
    integer :: a
  end type
  !$omp declare mapper(t :: v) map(v%a)
contains
  subroutine test_motion(x)
    type(t) :: x

    !$omp target enter data map(to: x)
    !$omp target update to(x)
    !$omp target update from(x)
    !$omp target update to(mapper(default): x)
    !$omp target exit data map(from: x)
  end subroutine test_motion
end module m

! CHECK-LABEL: func.func @_QMmPtest_motion(
! CHECK: %[[ENTER:.*]] = omp.map.info {{.*}} map_clauses(to) capture(ByRef) -> {{.*}} {name = "x"}
! CHECK: omp.target_enter_data map_entries(%[[ENTER]]
! CHECK: %[[UPTO:.*]] = omp.map.info {{.*}} map_clauses(to) capture(ByRef) -> {{.*}} {name = "x"}
! CHECK: omp.target_update map_entries(%[[UPTO]]
! CHECK: %[[UPFROM:.*]] = omp.map.info {{.*}} map_clauses(from) capture(ByRef) -> {{.*}} {name = "x"}
! CHECK: omp.target_update map_entries(%[[UPFROM]]
! CHECK: %[[UPDEFAULT:.*]] = omp.map.info {{.*}} map_clauses(to) capture(ByRef) mapper(@{{.*}}t_omp_default_mapper) -> {{.*}} {name = "x"}
! CHECK: omp.target_update map_entries(%[[UPDEFAULT]]
! CHECK: %[[EXIT:.*]] = omp.map.info {{.*}} map_clauses(from) capture(ByRef) -> {{.*}} {name = "x"}
! CHECK: omp.target_exit_data map_entries(%[[EXIT]]

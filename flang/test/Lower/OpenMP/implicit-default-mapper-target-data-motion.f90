! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s

module implicit_default_mapper_target_data_motion
  type :: t
    integer, allocatable :: a(:)
  end type
contains
  subroutine s(x)
    type(t) :: x

    !$omp target enter data map(to: x)
    !$omp target update to(x)
    !$omp target update from(x)
    !$omp target exit data map(from: x)
  end subroutine
end module

! CHECK: omp.declare_mapper @[[MAPPER:.*_omp_default_mapper]] :
! CHECK: omp.map.info {{.*}} map_clauses(implicit, tofrom, ref_ptee) {{.*}}name("")
! CHECK: omp.map.info {{.*}} map_clauses(attach, ref_ptee) {{.*}}name("")
! CHECK: omp.map.info {{.*}} map_clauses(implicit, tofrom) capture(ByRef) members(%{{.*}} : [0]
! CHECK: omp.declare_mapper.info map_entries(

! CHECK-LABEL: func.func @_QMimplicit_default_mapper_target_data_motionPs(
! CHECK: %[[ENTER_MAP:.*]] = omp.map.info {{.*}} map_clauses(to) capture(ByRef) mapper(@[[MAPPER]]) name("x")
! CHECK-NEXT: omp.target_enter_data map_entries(%[[ENTER_MAP]]
! CHECK: %[[UPDATE_TO_MAP:.*]] = omp.map.info {{.*}} map_clauses(to) capture(ByRef) mapper(@[[MAPPER]]) name("x")
! CHECK-NEXT: omp.target_update map_entries(%[[UPDATE_TO_MAP]]
! CHECK: %[[UPDATE_FROM_MAP:.*]] = omp.map.info {{.*}} map_clauses(from) capture(ByRef) mapper(@[[MAPPER]]) name("x")
! CHECK-NEXT: omp.target_update map_entries(%[[UPDATE_FROM_MAP]]
! CHECK: %[[EXIT_MAP:.*]] = omp.map.info {{.*}} map_clauses(from) capture(ByRef) mapper(@[[MAPPER]]) name("x")
! CHECK-NEXT: omp.target_exit_data map_entries(%[[EXIT_MAP]]

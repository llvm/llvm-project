! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s

module implicit_default_mapper_target_data_motion
  type :: t_enter_update_to_update_from_exit
    integer, allocatable :: a(:)
  end type

  type :: t_exit_update_to_update_from_enter
    integer, allocatable :: a(:)
  end type

  type :: t_update_to_update_from_enter_exit
    integer, allocatable :: a(:)
  end type

  type :: t_update_from_enter_exit_update_to
    integer, allocatable :: a(:)
  end type
contains
  subroutine enter_update_to_update_from_exit(x)
    type(t_enter_update_to_update_from_exit) :: x

    !$omp target enter data map(to: x)
    !$omp target update to(x)
    !$omp target update from(x)
    !$omp target exit data map(from: x)
  end subroutine

  subroutine exit_update_to_update_from_enter(x)
    type(t_exit_update_to_update_from_enter) :: x

    !$omp target exit data map(from: x)
    !$omp target update to(x)
    !$omp target update from(x)
    !$omp target enter data map(to: x)
  end subroutine

  subroutine update_to_update_from_enter_exit(x)
    type(t_update_to_update_from_enter_exit) :: x

    !$omp target update to(x)
    !$omp target update from(x)
    !$omp target enter data map(to: x)
    !$omp target exit data map(from: x)
  end subroutine

  subroutine update_from_enter_exit_update_to(x)
    type(t_update_from_enter_exit_update_to) :: x

    !$omp target update from(x)
    !$omp target enter data map(to: x)
    !$omp target exit data map(from: x)
    !$omp target update to(x)
  end subroutine
end module

! CHECK: omp.declare_mapper @[[UPDATE_FROM_ENTER_EXIT_UPDATE_TO_MAPPER:.*t_update_from_enter_exit_update_to_omp_default_mapper]] :
! CHECK: omp.map.info {{.*}} map_clauses(implicit, tofrom, ref_ptee) {{.*}}name("")
! CHECK: omp.map.info {{.*}} map_clauses(attach, ref_ptee) {{.*}}name("")
! CHECK: omp.map.info {{.*}} map_clauses(implicit, tofrom) capture(ByRef) members(%{{.*}} : [0]
! CHECK: omp.declare_mapper.info map_entries(
! CHECK: omp.declare_mapper @[[UPDATE_TO_UPDATE_FROM_ENTER_EXIT_MAPPER:.*t_update_to_update_from_enter_exit_omp_default_mapper]] :
! CHECK: omp.map.info {{.*}} map_clauses(implicit, tofrom, ref_ptee) {{.*}}name("")
! CHECK: omp.map.info {{.*}} map_clauses(attach, ref_ptee) {{.*}}name("")
! CHECK: omp.map.info {{.*}} map_clauses(implicit, tofrom) capture(ByRef) members(%{{.*}} : [0]
! CHECK: omp.declare_mapper.info map_entries(
! CHECK: omp.declare_mapper @[[EXIT_UPDATE_TO_UPDATE_FROM_ENTER_MAPPER:.*t_exit_update_to_update_from_enter_omp_default_mapper]] :
! CHECK: omp.map.info {{.*}} map_clauses(implicit, tofrom, ref_ptee) {{.*}}name("")
! CHECK: omp.map.info {{.*}} map_clauses(attach, ref_ptee) {{.*}}name("")
! CHECK: omp.map.info {{.*}} map_clauses(implicit, tofrom) capture(ByRef) members(%{{.*}} : [0]
! CHECK: omp.declare_mapper.info map_entries(
! CHECK: omp.declare_mapper @[[ENTER_UPDATE_TO_UPDATE_FROM_EXIT_MAPPER:.*t_enter_update_to_update_from_exit_omp_default_mapper]] :
! CHECK: omp.map.info {{.*}} map_clauses(implicit, tofrom, ref_ptee) {{.*}}name("")
! CHECK: omp.map.info {{.*}} map_clauses(attach, ref_ptee) {{.*}}name("")
! CHECK: omp.map.info {{.*}} map_clauses(implicit, tofrom) capture(ByRef) members(%{{.*}} : [0]
! CHECK: omp.declare_mapper.info map_entries(

! CHECK-LABEL: func.func @_QMimplicit_default_mapper_target_data_motionPenter_update_to_update_from_exit(
! CHECK: %[[ENTER_UPDATE_TO_UPDATE_FROM_EXIT_ENTER_MAP:.*]] = omp.map.info {{.*}} map_clauses(to) capture(ByRef) mapper(@[[ENTER_UPDATE_TO_UPDATE_FROM_EXIT_MAPPER]]) name("x")
! CHECK-NEXT: omp.target_enter_data map_entries(%[[ENTER_UPDATE_TO_UPDATE_FROM_EXIT_ENTER_MAP]]
! CHECK: %[[ENTER_UPDATE_TO_UPDATE_FROM_EXIT_UPDATE_TO_MAP:.*]] = omp.map.info {{.*}} map_clauses(to) capture(ByRef) mapper(@[[ENTER_UPDATE_TO_UPDATE_FROM_EXIT_MAPPER]]) name("x")
! CHECK-NEXT: omp.target_update map_entries(%[[ENTER_UPDATE_TO_UPDATE_FROM_EXIT_UPDATE_TO_MAP]]
! CHECK: %[[ENTER_UPDATE_TO_UPDATE_FROM_EXIT_UPDATE_FROM_MAP:.*]] = omp.map.info {{.*}} map_clauses(from) capture(ByRef) mapper(@[[ENTER_UPDATE_TO_UPDATE_FROM_EXIT_MAPPER]]) name("x")
! CHECK-NEXT: omp.target_update map_entries(%[[ENTER_UPDATE_TO_UPDATE_FROM_EXIT_UPDATE_FROM_MAP]]
! CHECK: %[[ENTER_UPDATE_TO_UPDATE_FROM_EXIT_EXIT_MAP:.*]] = omp.map.info {{.*}} map_clauses(from) capture(ByRef) mapper(@[[ENTER_UPDATE_TO_UPDATE_FROM_EXIT_MAPPER]]) name("x")
! CHECK-NEXT: omp.target_exit_data map_entries(%[[ENTER_UPDATE_TO_UPDATE_FROM_EXIT_EXIT_MAP]]

! CHECK-LABEL: func.func @_QMimplicit_default_mapper_target_data_motionPexit_update_to_update_from_enter(
! CHECK: %[[EXIT_UPDATE_TO_UPDATE_FROM_ENTER_EXIT_MAP:.*]] = omp.map.info {{.*}} map_clauses(from) capture(ByRef) mapper(@[[EXIT_UPDATE_TO_UPDATE_FROM_ENTER_MAPPER]]) name("x")
! CHECK-NEXT: omp.target_exit_data map_entries(%[[EXIT_UPDATE_TO_UPDATE_FROM_ENTER_EXIT_MAP]]
! CHECK: %[[EXIT_UPDATE_TO_UPDATE_FROM_ENTER_UPDATE_TO_MAP:.*]] = omp.map.info {{.*}} map_clauses(to) capture(ByRef) mapper(@[[EXIT_UPDATE_TO_UPDATE_FROM_ENTER_MAPPER]]) name("x")
! CHECK-NEXT: omp.target_update map_entries(%[[EXIT_UPDATE_TO_UPDATE_FROM_ENTER_UPDATE_TO_MAP]]
! CHECK: %[[EXIT_UPDATE_TO_UPDATE_FROM_ENTER_UPDATE_FROM_MAP:.*]] = omp.map.info {{.*}} map_clauses(from) capture(ByRef) mapper(@[[EXIT_UPDATE_TO_UPDATE_FROM_ENTER_MAPPER]]) name("x")
! CHECK-NEXT: omp.target_update map_entries(%[[EXIT_UPDATE_TO_UPDATE_FROM_ENTER_UPDATE_FROM_MAP]]
! CHECK: %[[EXIT_UPDATE_TO_UPDATE_FROM_ENTER_ENTER_MAP:.*]] = omp.map.info {{.*}} map_clauses(to) capture(ByRef) mapper(@[[EXIT_UPDATE_TO_UPDATE_FROM_ENTER_MAPPER]]) name("x")
! CHECK-NEXT: omp.target_enter_data map_entries(%[[EXIT_UPDATE_TO_UPDATE_FROM_ENTER_ENTER_MAP]]

! CHECK-LABEL: func.func @_QMimplicit_default_mapper_target_data_motionPupdate_to_update_from_enter_exit(
! CHECK: %[[UPDATE_TO_UPDATE_FROM_ENTER_EXIT_UPDATE_TO_MAP:.*]] = omp.map.info {{.*}} map_clauses(to) capture(ByRef) mapper(@[[UPDATE_TO_UPDATE_FROM_ENTER_EXIT_MAPPER]]) name("x")
! CHECK-NEXT: omp.target_update map_entries(%[[UPDATE_TO_UPDATE_FROM_ENTER_EXIT_UPDATE_TO_MAP]]
! CHECK: %[[UPDATE_TO_UPDATE_FROM_ENTER_EXIT_UPDATE_FROM_MAP:.*]] = omp.map.info {{.*}} map_clauses(from) capture(ByRef) mapper(@[[UPDATE_TO_UPDATE_FROM_ENTER_EXIT_MAPPER]]) name("x")
! CHECK-NEXT: omp.target_update map_entries(%[[UPDATE_TO_UPDATE_FROM_ENTER_EXIT_UPDATE_FROM_MAP]]
! CHECK: %[[UPDATE_TO_UPDATE_FROM_ENTER_EXIT_ENTER_MAP:.*]] = omp.map.info {{.*}} map_clauses(to) capture(ByRef) mapper(@[[UPDATE_TO_UPDATE_FROM_ENTER_EXIT_MAPPER]]) name("x")
! CHECK-NEXT: omp.target_enter_data map_entries(%[[UPDATE_TO_UPDATE_FROM_ENTER_EXIT_ENTER_MAP]]
! CHECK: %[[UPDATE_TO_UPDATE_FROM_ENTER_EXIT_EXIT_MAP:.*]] = omp.map.info {{.*}} map_clauses(from) capture(ByRef) mapper(@[[UPDATE_TO_UPDATE_FROM_ENTER_EXIT_MAPPER]]) name("x")
! CHECK-NEXT: omp.target_exit_data map_entries(%[[UPDATE_TO_UPDATE_FROM_ENTER_EXIT_EXIT_MAP]]

! CHECK-LABEL: func.func @_QMimplicit_default_mapper_target_data_motionPupdate_from_enter_exit_update_to(
! CHECK: %[[UPDATE_FROM_ENTER_EXIT_UPDATE_TO_UPDATE_FROM_MAP:.*]] = omp.map.info {{.*}} map_clauses(from) capture(ByRef) mapper(@[[UPDATE_FROM_ENTER_EXIT_UPDATE_TO_MAPPER]]) name("x")
! CHECK-NEXT: omp.target_update map_entries(%[[UPDATE_FROM_ENTER_EXIT_UPDATE_TO_UPDATE_FROM_MAP]]
! CHECK: %[[UPDATE_FROM_ENTER_EXIT_UPDATE_TO_ENTER_MAP:.*]] = omp.map.info {{.*}} map_clauses(to) capture(ByRef) mapper(@[[UPDATE_FROM_ENTER_EXIT_UPDATE_TO_MAPPER]]) name("x")
! CHECK-NEXT: omp.target_enter_data map_entries(%[[UPDATE_FROM_ENTER_EXIT_UPDATE_TO_ENTER_MAP]]
! CHECK: %[[UPDATE_FROM_ENTER_EXIT_UPDATE_TO_EXIT_MAP:.*]] = omp.map.info {{.*}} map_clauses(from) capture(ByRef) mapper(@[[UPDATE_FROM_ENTER_EXIT_UPDATE_TO_MAPPER]]) name("x")
! CHECK-NEXT: omp.target_exit_data map_entries(%[[UPDATE_FROM_ENTER_EXIT_UPDATE_TO_EXIT_MAP]]
! CHECK: %[[UPDATE_FROM_ENTER_EXIT_UPDATE_TO_UPDATE_TO_MAP:.*]] = omp.map.info {{.*}} map_clauses(to) capture(ByRef) mapper(@[[UPDATE_FROM_ENTER_EXIT_UPDATE_TO_MAPPER]]) name("x")
! CHECK-NEXT: omp.target_update map_entries(%[[UPDATE_FROM_ENTER_EXIT_UPDATE_TO_UPDATE_TO_MAP]]

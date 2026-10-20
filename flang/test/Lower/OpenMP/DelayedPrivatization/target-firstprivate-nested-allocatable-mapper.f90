! Test that firstprivate target privatization of an array of derived types
! generates implicit declare mappers when the array element type contains an
! array of another derived type that also contains an allocatable member.
!
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

subroutine target_firstprivate_nested_allocatable_mapper
  implicit none

  type :: inner_type
    integer, allocatable :: values(:)
  end type inner_type

  type :: outer_type
    type(inner_type) :: children(4)
  end type outer_type

  type(outer_type) :: objs(4)

  !$omp target firstprivate(objs)
    if (allocated(objs(1)%children(1)%values)) then
      objs(1)%children(1)%values(1) = 42
    end if
  !$omp end target
end subroutine target_firstprivate_nested_allocatable_mapper

! CHECK:       omp.declare_mapper @[[INNER_MAPPER:.*inner_type_omp_default_mapper]] : !fir.type<{{.*}}Tinner_type{{.*}}>
! CHECK:       %[[INNER_VALUES_MAP:.*]] = omp.map.info {{.*}}map_clauses(implicit, tofrom, ref_ptee)
! CHECK:       %[[INNER_VALUES_ATTACH:.*]] = omp.map.info {{.*}}map_clauses(attach, ref_ptee)
! CHECK:       %[[INNER_PARENT_MAP:.*]] = omp.map.info {{.*}}map_clauses(implicit, tofrom){{.*}}members(%[[INNER_VALUES_MAP]] : [0] :

! CHECK:       omp.declare_mapper @[[OUTER_MAPPER:.*outer_type_omp_default_mapper]] : !fir.type<{{.*}}Touter_type{{.*}}>
! CHECK:       %[[OUTER_CHILDREN_MAP:.*]] = omp.map.info {{.*}}map_clauses(implicit, tofrom){{.*}}mapper(@[[INNER_MAPPER]])
! CHECK:       %[[OUTER_PARENT_MAP:.*]] = omp.map.info {{.*}}map_clauses(implicit, tofrom){{.*}}members(%[[OUTER_CHILDREN_MAP]] : [0] :

! CHECK:       omp.private {type = firstprivate} @{{.*}}objs_firstprivate{{.*}}outer_type : !fir.box<!fir.array<4x!fir.type<

! CHECK-LABEL: func.func @_QPtarget_firstprivate_nested_allocatable_mapper()
! CHECK:       %[[OBJS_DATA_MAP:.*]] = omp.map.info {{.*}}map_clauses(tofrom){{.*}}mapper(@[[OUTER_MAPPER]])
! CHECK:       %[[OBJS_DESC_MAP:.*]] = omp.map.info {{.*}}map_clauses(always, to){{.*}}members(%[[OBJS_DATA_MAP]] : [0] :
! CHECK:       omp.target {{.*}}map_entries(%[[OBJS_DESC_MAP]] -> {{.*}}, {{.*}}, %[[OBJS_DATA_MAP]] ->{{.*}}private(@{{.*}}objs_firstprivate{{.*}}outer_type {{.*}}[map_idx=0]

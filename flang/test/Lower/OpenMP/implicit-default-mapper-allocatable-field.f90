! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s

program p
  implicit none
  type t
    integer, allocatable :: a(:)
  end type
  type(t) :: x

  allocate(x%a(1))
  x%a = 0

  !$omp target
    x%a(1) = 42
  !$omp end target

  if (x%a(1) /= 42) error stop
end program

! CHECK: omp.declare_mapper @[[MAPPER:_QQFt_omp_default_mapper]] : [[TYPE:!fir\.type<_QFTt\{a:!fir\.box<!fir\.heap<!fir\.array<\?xi32>>>\}>]] {
! CHECK: ^bb0(%[[ARG:.*]]: !fir.ref<[[TYPE]]>):
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ARG]]
! CHECK: %[[COORD:.*]] = fir.coordinate_of %[[DECL]]#0, a
! CHECK: %[[BOUNDS:.*]] = omp.map.bounds
! CHECK: %[[BOX_OFF:.*]] = fir.box_offset %[[COORD]] base_addr
! CHECK: %[[MEMBER_PTR:.*]] = omp.map.info var_ptr(%[[COORD]] {{.*}}) map_clauses(implicit, tofrom, ref_ptee) capture(ByRef) var_ptr_ptr(%[[BOX_OFF]] {{.*}}) bounds(%[[BOUNDS]]) name("")
! CHECK: %[[ATTACH:.*]] = omp.map.info var_ptr(%[[COORD]] {{.*}}) map_clauses(attach, ref_ptee) capture(ByRef) var_ptr_ptr(%[[BOX_OFF]] {{.*}}) bounds(%[[BOUNDS]]) name("")
! CHECK: %[[PARENT:.*]] = omp.map.info var_ptr(%[[DECL]]#1 {{.*}}) map_clauses(implicit, tofrom) capture(ByRef) members(%[[MEMBER_PTR]] : [0] {{.*}}) name("")
! CHECK: omp.declare_mapper.info map_entries(%[[PARENT]], %[[MEMBER_PTR]], %[[ATTACH]]

! CHECK: %[[MAP:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<[[TYPE]]>, [[TYPE]]) map_clauses(implicit, tofrom) capture(ByRef) mapper(@[[MAPPER]]) name("x")
! CHECK: omp.target {{.*}}map_entries(%[[MAP]] -> %{{.*}} : !fir.ref<[[TYPE]]>)

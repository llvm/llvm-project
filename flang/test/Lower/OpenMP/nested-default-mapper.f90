! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s

program main
  implicit none

  type nested_t
    integer, allocatable :: y(:)
  end type nested_t

  !$omp declare mapper(nested_t :: n) map(n%y)

  type typ_t
    integer, allocatable :: x(:)
    type(nested_t) :: nested
  end type typ_t

  !$omp declare mapper(typ_t :: t) map(t%x, t%nested)

  type(typ_t) :: typ

  allocate(typ%x(3), source=1)
  allocate(typ%nested%y(3), source=42)

  !$omp target map(tofrom: typ)
    typ%x(1) = 999
    typ%nested%y(1) = -555
  !$omp end target
end program main

subroutine recursive_mapper
  implicit none

  type node_t
    integer, allocatable :: payload(:)
    type(node_t), allocatable :: next
  end type node_t

  type wrapper_t
    type(node_t), allocatable :: head
  end type wrapper_t

  type(wrapper_t) :: w

  !$omp target map(tofrom: w)
    w%head%next%next%payload(1) = 7
  !$omp end target
end subroutine recursive_mapper

subroutine mutual_mapper
  implicit none

  type a_t
    type(b_t), allocatable :: b
  end type a_t

  type b_t
    type(a_t), allocatable :: a
    integer, allocatable :: payload(:)
  end type b_t

  type(a_t) :: a

  !$omp target map(tofrom: a)
    a%b%a%b%payload(1) = 11
  !$omp end target
end subroutine mutual_mapper

! CHECK-LABEL: omp.declare_mapper @_QQFmutual_mapper_QFmutual_mapperTa_t_omp_default_mapper
! CHECK: omp.map.info {{.*}} mapper(@_QFmutual_mapperTb_t_omp_default_mapper)
! CHECK-LABEL: omp.declare_mapper @_QFmutual_mapperTb_t_omp_default_mapper
! CHECK: omp.map.info {{.*}} mapper(@_QQFmutual_mapper_QFmutual_mapperTa_t_omp_default_mapper)
! CHECK-LABEL: omp.declare_mapper @_QQFmutual_mappera_t_omp_default_mapper
! CHECK: omp.map.info {{.*}} mapper(@_QFmutual_mapperTb_t_omp_default_mapper)

! CHECK-LABEL: omp.declare_mapper @_QFrecursive_mapperTnode_t_omp_default_mapper
! CHECK: omp.map.info {{.*}} mapper(@_QFrecursive_mapperTnode_t_omp_default_mapper)
! CHECK-LABEL: omp.declare_mapper @_QQFrecursive_mapperwrapper_t_omp_default_mapper
! CHECK: omp.map.info {{.*}} mapper(@_QFrecursive_mapperTnode_t_omp_default_mapper)

! CHECK-LABEL: omp.declare_mapper @_QQFtyp_t_omp_default_mapper
! CHECK: omp.map.info {{.*}} map_clauses(tofrom) capture(ByRef) mapper(@_QQFnested_t_omp_default_mapper) -> {{.*}} {name = "t%nested"}
! CHECK-LABEL: omp.declare_mapper @_QQFnested_t_omp_default_mapper

! CHECK-LABEL: func.func @_QQmain
! CHECK-NOT: implicit_map
! CHECK: %[[TYP_MAP:.*]] = omp.map.info {{.*}} mapper(@_QQFtyp_t_omp_default_mapper){{.*}} {name = "typ"}
! CHECK-NEXT: omp.target kernel_type(generic) map_entries(%[[TYP_MAP]] -> %{{[^,]*}} : {{.*}}) {

! CHECK-LABEL: func.func @_QPrecursive_mapper
! CHECK-NOT: implicit_map
! CHECK: %[[RECURSIVE_MAP:.*]] = omp.map.info {{.*}} mapper(@_QQFrecursive_mapperwrapper_t_omp_default_mapper){{.*}} {name = "w"}
! CHECK-NEXT: omp.target kernel_type(generic) map_entries(%[[RECURSIVE_MAP]] -> %{{[^,]*}} : {{.*}}) {

! CHECK-LABEL: func.func @_QPmutual_mapper
! CHECK-NOT: implicit_map
! CHECK: %[[MUTUAL_MAP:.*]] = omp.map.info {{.*}} mapper(@_QQFmutual_mappera_t_omp_default_mapper){{.*}} {name = "a"}
! CHECK-NEXT: omp.target kernel_type(generic) map_entries(%[[MUTUAL_MAP]] -> %{{[^,]*}} : {{.*}}) {

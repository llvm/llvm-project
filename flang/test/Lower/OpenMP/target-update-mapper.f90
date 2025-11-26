! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s

! Test mapper usage in target update to/from clauses

program target_update_mapper
  implicit none

  integer, parameter :: n = 4

  type :: typ
    integer, allocatable :: a(:)
    integer, allocatable :: b(:)
  end type typ

  !$omp declare mapper(custom: typ :: t) map(t%a)

  ! CHECK-LABEL: omp.declare_mapper @_QQFcustom : !fir.type<_QFTtyp{a:!fir.box<!fir.heap<!fir.array<?xi32>>>,b:!fir.box<!fir.heap<!fir.array<?xi32>>>}>

  type(typ) :: t

  allocate(t%a(n), source=1)
  allocate(t%b(n), source=2)

  !$omp target enter data map(alloc: t)

  ! Test target update to with custom mapper
  ! CHECK: %[[T_VAR:.*]] = fir.declare %{{.*}} {uniq_name = "_QFtarget_update_mapperEt"} : (!fir.ref<!fir.type<_QFTtyp{a:!fir.box<!fir.heap<!fir.array<?xi32>>>,b:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.ref<!fir.type<_QFTtyp{a:!fir.box<!fir.heap<!fir.array<?xi32>>>,b:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
  ! CHECK: %[[MAP_INFO:.*]] = omp.map.info var_ptr(%[[T_VAR]] : {{.*}}, {{.*}}) map_clauses(to) capture(ByRef) mapper(@_QQFcustom) -> {{.*}}
  ! CHECK: omp.target_update motion_entries(%[[MAP_INFO]] : {{.*}})
  t%a = 42
  !$omp target update to(mapper(custom): t)

  !$omp target
    t%a(:) = t%a(:) / 2
    t%b(:) = -1
  !$omp end target

  ! Test target update from with custom mapper
  ! CHECK: %[[MAP_INFO2:.*]] = omp.map.info var_ptr(%{{.*}} : {{.*}}, {{.*}}) map_clauses(from) capture(ByRef) mapper(@_QQFcustom) -> {{.*}}
  ! CHECK: omp.target_update motion_entries(%[[MAP_INFO2]] : {{.*}})
  !$omp target update from(mapper(custom): t)

  ! Test target update to with default mapper
  ! CHECK: %[[MAP_INFO3:.*]] = omp.map.info var_ptr(%{{.*}} : {{.*}}, {{.*}}) map_clauses(to) capture(ByRef) mapper(@_QQFtyp_omp_default_mapper) -> {{.*}}
  ! CHECK: omp.target_update motion_entries(%[[MAP_INFO3]] : {{.*}})
  !$omp target update to(mapper(default): t)

  !$omp target exit data map(delete: t)
  deallocate(t%a)
  deallocate(t%b)

end program target_update_mapper

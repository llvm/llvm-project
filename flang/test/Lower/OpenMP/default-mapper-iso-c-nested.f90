! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s

! Test that an implicit default declare mapper is synthesized for ISO C
! interoperable types.

program test_iso_c_nested_mapper
  use iso_c_binding, only : c_ptr
  implicit none

  type :: dtype
    integer     :: id
    type(c_ptr) :: ptr
  end type dtype

  type(dtype), allocatable :: arr(:)

  allocate(arr(10))

  !$omp target map(tofrom: arr)
    arr(1)%id = 1
  !$omp end target
end program test_iso_c_nested_mapper

subroutine map_cptr_direct
  use iso_c_binding, only : c_ptr
  implicit none
  type(c_ptr) :: p

  !$omp target map(tofrom: p)
  !$omp end target
end subroutine map_cptr_direct


! CHECK-LABEL: omp.declare_mapper @_QQM__fortran_builtinsc_ptr_omp_default_mapper
! CHECK: omp.map.info var_ptr({{.*}}) map_clauses(implicit, tofrom){{.*}}name("")
! CHECK: omp.map.info var_ptr({{.*}}) map_clauses(implicit){{.*}}members({{.*}}){{.*}}name("") partial_map(true)
! CHECK: omp.declare_mapper.info map_entries(

! CHECK-LABEL: func.func @_QQmain
! CHECK: %[[ARR_DATA:.*]] = omp.map.info var_ptr(%{{.*}}){{.*}}map_clauses(tofrom){{.*}}var_ptr_ptr(%{{.*}}){{.*}}name("")
! CHECK: %[[ARR_DESC:.*]] = omp.map.info var_ptr(%{{.*}}){{.*}}map_clauses(always, to){{.*}}members(%[[ARR_DATA]] : [0] :{{.*}}){{.*}}name("arr")
! CHECK: %[[ARR_ATTACH:.*]] = omp.map.info var_ptr(%{{.*}}){{.*}}map_clauses(attach, ref_ptr, ref_ptee){{.*}}name("arr")
! CHECK: omp.target {{.*}}map_entries(%[[ARR_DESC]] -> %{{[^,]*}}, %[[ARR_ATTACH]] -> %{{[^,]*}}, %[[ARR_DATA]] -> %{{[^,]*}} :

! CHECK-LABEL: func.func @_QPmap_cptr_direct
! CHECK: %[[P_MAP:.*]] = omp.map.info var_ptr(%{{.*}}){{.*}}map_clauses(tofrom){{.*}}mapper(@_QQM__fortran_builtinsc_ptr_omp_default_mapper){{.*}}name("p")
! CHECK: omp.target {{.*}}map_entries(%[[P_MAP]] -> %{{[^,]*}} :

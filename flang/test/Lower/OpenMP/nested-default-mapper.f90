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

! CHECK-LABEL: omp.declare_mapper @_QQFtyp_t_omp_default_mapper
! CHECK: omp.map.info {{.*}} map_clauses(tofrom) capture(ByRef) mapper(@_QQFnested_t_omp_default_mapper) -> {{.*}} {name = "t%nested"}
! CHECK-LABEL: omp.declare_mapper @_QQFnested_t_omp_default_mapper

! CHECK-LABEL: func.func @_QQmain
! CHECK-NOT: implicit_map
! CHECK: %[[TYP_MAP:.*]] = omp.map.info {{.*}} mapper(@_QQFtyp_t_omp_default_mapper){{.*}} {name = "typ"}
! CHECK-NEXT: omp.target kernel_type(generic) map_entries(%[[TYP_MAP]] -> %{{[^,]*}} : {{.*}}) {

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s
!
! Check that nested implicit default mapper generation for PDT components reuses
! an explicit default declare mapper(default:) when available for that exact
! instantiated type.

module m
  type :: pdt(k)
    integer, kind :: k
    real(k) :: x
  end type

  !$omp declare mapper(default: pdt(4) :: v) map(tofrom: v%x)

  type :: holder4
    type(pdt(4)), allocatable :: a(:)
  end type

  type :: holder8
    type(pdt(8)), allocatable :: a(:)
  end type
contains
  subroutine s(h4, h8)
    type(holder4) :: h4
    type(holder8) :: h8
    !$omp target data map(h4, h8)
    !$omp end target data
  end subroutine
end module

! Explicit mapper for pdt(4) keeps the historical kindless symbol spelling.
! CHECK-DAG: omp.declare_mapper @_QQMmpdt_omp_default_mapper : !fir.type<_QMmTpdtK4

! Implicit mapper for pdt(8) remains distinct and kind-qualified.
! CHECK-DAG: omp.declare_mapper @_QQMmpdtK8_omp_default_mapper : !fir.type<_QMmTpdtK8

! holder4 nested mapping must reuse the explicit pdt(4) mapper, not synthesize
! a pdtK4 implicit default mapper.
! CHECK-DAG: mapper(@_QQMmpdt_omp_default_mapper)
! CHECK-DAG: mapper(@_QQMmpdtK8_omp_default_mapper)
! CHECK-NOT: omp.declare_mapper @_QQMmpdtK4_omp_default_mapper

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s
!
! Check that implicit default mapper generation does not alias parameterized
! derived type instantiations with different kind parameters.
! Before the fix both pdt(4) and pdt(8) nested mappings could bind to the same
! mapper symbol due to kind-less name canonicalization.

module m
  type :: pdt(k)
    integer, kind :: k
    real(k) :: x
  end type

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

! CHECK-DAG: omp.declare_mapper @_QQMmpdtK4_omp_default_mapper
! CHECK-DAG: omp.declare_mapper @_QQMmpdtK8_omp_default_mapper
! CHECK-DAG: omp.declare_mapper @_QQMmholder4_omp_default_mapper
! CHECK-DAG: omp.declare_mapper @_QQMmholder8_omp_default_mapper
! CHECK-DAG: mapper(@_QQMmpdtK4_omp_default_mapper)
! CHECK-DAG: mapper(@_QQMmpdtK8_omp_default_mapper)

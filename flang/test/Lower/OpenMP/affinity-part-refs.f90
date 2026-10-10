! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s | \
! RUN:   FileCheck %s

! These locators must not be rejected by the earlier-part section guard.

subroutine scalar_parent(a, b, v)
  type t
    integer :: field(4)
  end type
  type(t) :: a(8)
  integer :: b(100), v(8)
  !$omp task affinity(iterator(i=1:2): a(i)%field(1))
  !$omp end task
end
! CHECK-LABEL: func.func @_QPscalar_parent(
! CHECK: omp.iterator
! CHECK: omp.affinity_entry
! CHECK: omp.task affinity

subroutine final_section(a, b, v)
  type t
    integer :: field(4)
  end type
  type(t) :: a(8)
  integer :: b(100), v(8)
  !$omp task affinity(iterator(i=1:2): a(1)%field(i:i+1))
  !$omp end task
end
! CHECK-LABEL: func.func @_QPfinal_section(
! CHECK: omp.iterator
! CHECK: omp.affinity_entry
! CHECK: omp.task affinity

subroutine ordinary_expression(a, b, v)
  type t
    integer :: field(4)
  end type
  type(t) :: a(8)
  integer :: b(100), v(8)
  !$omp task affinity(b(sum(v(::2))))
  !$omp end task
end
! CHECK-LABEL: func.func @_QPordinary_expression(
! CHECK: omp.affinity_entry
! CHECK: omp.task affinity

subroutine iterator_expression(a, b, v)
  type t
    integer :: field(4)
  end type
  type(t) :: a(8)
  integer :: b(100), v(8)
  !$omp task affinity(iterator(i=1:2): b(sum(v(::2))+i))
  !$omp end task
end
! CHECK-LABEL: func.func @_QPiterator_expression(
! CHECK: omp.iterator
! CHECK: omp.affinity_entry
! CHECK: omp.task affinity

subroutine parent_expression(a, b, v)
  type t
    integer :: field(4)
  end type
  type(t) :: a(8)
  integer :: b(100), v(8)
  !$omp task affinity(iterator(i=1:2): a(sum(v(::2)))%field(i))
  !$omp end task
end
! CHECK-LABEL: func.func @_QPparent_expression(
! CHECK: omp.iterator
! CHECK: omp.affinity_entry
! CHECK: omp.task affinity

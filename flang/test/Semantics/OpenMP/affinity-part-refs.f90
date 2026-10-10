! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52

subroutine earlier_part_strides(a, k)
  type inner
    integer :: field(4)
  end type
  type outer
    type(inner) :: nested
  end type
  type(outer) :: a(8)
  integer :: k

  !ERROR: 'a' in AFFINITY clause must not specify a stride
  !$omp task affinity(a(1:8:1)%nested%field(1))
  !$omp end task
  !ERROR: 'a' in AFFINITY clause must not specify a stride
  !$omp task affinity(iterator(i=1:2): a(1:8:1)%nested%field(i))
  !$omp end task
  !ERROR: 'a' in AFFINITY clause must not specify a stride
  !$omp task affinity(a(1:8:2)%nested%field(1))
  !$omp end task
  !ERROR: 'a' in AFFINITY clause must not specify a stride
  !$omp task affinity(iterator(i=1:2): a(1:8:2)%nested%field(i))
  !$omp end task
  !ERROR: 'a' in AFFINITY clause must not specify a stride
  !$omp task affinity(a(1:8:k)%nested%field(1))
  !$omp end task
  !ERROR: 'a' in AFFINITY clause must not specify a stride
  !$omp task affinity(iterator(i=1:2): a(1:8:k)%nested%field(i))
  !$omp end task
  !ERROR: 'a' in AFFINITY clause must not specify a stride
  !$omp task affinity(a(::2)%nested%field(1))
  !$omp end task
  !ERROR: 'a' in AFFINITY clause must not specify a stride
  !$omp task affinity(iterator(i=1:2): a(::2)%nested%field(i))
  !$omp end task
end subroutine

subroutine intermediate_part_stride(a)
  type inner
    integer :: field(4)
  end type
  type outer
    type(inner) :: nested(8)
  end type
  type(outer) :: a
  !ERROR: 'nested' in AFFINITY clause must not specify a stride
  !$omp task affinity(iterator(i=1:2): a%nested(::2)%field(i))
  !$omp end task
end subroutine

! A scalar final subscript satisfies the last-part-ref restriction.
! Sections inside scalar expressions are not sections of the locator.
subroutine valid_part_refs(a, b, v)
  type t
    integer :: field(4)
  end type
  type(t) :: a(8)
  integer :: b(100), v(8)
  !$omp task affinity(a%field(1))
  !$omp end task
  !$omp task affinity(iterator(i=1:2): a%field(i))
  !$omp end task
  !$omp task affinity(a(1:2)%field(1))
  !$omp end task
  !$omp task affinity(iterator(i=1:2): a(1:2)%field(i))
  !$omp end task
  !$omp task affinity(b(sum(v(::2))))
  !$omp end task
  !$omp task affinity(iterator(i=1:2): b(sum(v(::2))+i))
  !$omp end task
  !$omp task affinity(iterator(i=1:2): a(sum(v(::2)))%field(i))
  !$omp end task
end subroutine

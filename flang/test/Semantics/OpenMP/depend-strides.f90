! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52

subroutine depend_strides(b, k)
  integer :: b(8,8), k
  !ERROR: 'b' in DEPEND clause must not specify a stride
  !$omp task depend(in: b(1:8:1,1))
  !$omp end task
  !ERROR: 'b' in DEPEND clause must not specify a stride
  !$omp task depend(in: b(1,1:8:2))
  !$omp end task
  !ERROR: 'b' in DEPEND clause must not specify a stride
  !$omp task depend(in: b(1:8:k,1))
  !$omp end task
  !ERROR: 'b' in DEPEND clause must not specify a stride
  !$omp task depend(in: b(1,::2))
  !$omp end task
  !ERROR: 'b' in DEPEND clause must not specify a stride
  !$omp task depend(iterator(i=1:2), in: b(i,1:8:k))
  !$omp end task
  ! One diagnostic per part reference, even with multiple strides.
  !ERROR: 'b' in DEPEND clause must not specify a stride
  !$omp task depend(in: b(1:8:1,1:8:2))
  !$omp end task
end subroutine

subroutine earlier_part_strides(a, k)
  type inner
    integer :: field(4)
  end type
  type outer
    type(inner) :: nested(8)
  end type
  type(outer) :: a(8)
  integer :: k
  !ERROR: 'a' in DEPEND clause must not specify a stride
  !$omp task depend(in: a(1:8:1)%nested(1)%field(1))
  !$omp end task
  !ERROR: 'a' in DEPEND clause must not specify a stride
  !$omp task depend(iterator(i=1:2), in: a(::2)%nested(1)%field(i))
  !$omp end task
  !ERROR: 'nested' in DEPEND clause must not specify a stride
  !$omp task depend(iterator(i=1:2), in: a(1)%nested(::k)%field(i))
  !$omp end task
end subroutine

subroutine valid_sections(a, b, v)
  type t
    integer :: field(4)
  end type
  type(t) :: a(8)
  integer :: b(100), v(8)
  !$omp task depend(in: a(1:2)%field(1))
  !$omp end task
  !$omp task depend(iterator(i=1:2), in: a(1:2)%field(i))
  !$omp end task
  !$omp task depend(in: b(sum(v(::2))))
  !$omp end task
  !$omp task depend(iterator(i=1:2:1), in: b(sum(v(::2))+i))
  !$omp end task
  !$omp task depend(iterator(i=1:2), in: a(sum(v(::2)))%field(i))
  !$omp end task
end subroutine

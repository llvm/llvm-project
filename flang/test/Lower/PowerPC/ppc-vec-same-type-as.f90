! RUN: %flang -o %t %s && %t
! REQUIRES: target=powerpc{{.*}}-{{.*}}-aix{{.*}}
!
! Test that same_type_as works correctly when a class(*) allocatable holds an
! IBM vector extension type:
!   - same_type_as(x, x) must be .true. for integer, unsigned, and real vectors
!   - same_type_as(x, y) must be .false. across distinct vector types
!   - allocate with source= and mold= from a plain local vector variable

subroutine test_allocate_source()
  vector(integer(4))  :: vi(2)
  vector(unsigned(4)) :: vu(2)
  vector(real(4))     :: vr(2)
  class(*), allocatable :: x(:)

  allocate(x(2), source=vi)
  if (.not. same_type_as(x, x)) error stop "FAIL: source= vector(integer(4))"
  deallocate(x)

  allocate(x(2), source=vu)
  if (.not. same_type_as(x, x)) error stop "FAIL: source= vector(unsigned(4))"
  deallocate(x)

  allocate(x(2), source=vr)
  if (.not. same_type_as(x, x)) error stop "FAIL: source= vector(real(4))"
  deallocate(x)
end subroutine

subroutine test_allocate_mold()
  vector(integer(4))  :: vi(2)
  vector(unsigned(4)) :: vu(2)
  vector(real(4))     :: vr(2)
  class(*), allocatable :: x(:)

  allocate(x(2), mold=vi)
  if (.not. same_type_as(x, x)) error stop "FAIL: mold= vector(integer(4))"
  deallocate(x)

  allocate(x(2), mold=vu)
  if (.not. same_type_as(x, x)) error stop "FAIL: mold= vector(unsigned(4))"
  deallocate(x)

  allocate(x(2), mold=vr)
  if (.not. same_type_as(x, x)) error stop "FAIL: mold= vector(real(4))"
  deallocate(x)
end subroutine

subroutine test_assign()
  vector(integer(4))  :: vi(2)
  vector(unsigned(4)) :: vu(2)
  vector(real(4))     :: vr(2)
  class(*), allocatable :: x(:)

  allocate(x(2), source=vi)
  x = vr
  if (.not. same_type_as(x, x)) error stop "FAIL: assign vector(real(4))"
  x = vu
  if (.not. same_type_as(x, x)) error stop "FAIL: assign vector(unsigned(4))"
  deallocate(x)
end subroutine

subroutine test_cross_type()
  vector(integer(4))  :: vi(2)
  vector(unsigned(4)) :: vu(2)
  vector(real(4))     :: vr(2)
  class(*), allocatable :: xi(:), xu(:), xr(:)

  allocate(xi(2), source=vi)
  allocate(xu(2), source=vu)
  allocate(xr(2), source=vr)

  ! integer vs real
  if (same_type_as(xi, xr)) &
      error stop "FAIL: vector(integer(4)) same_type_as vector(real(4))"
  if (same_type_as(xr, xi)) &
      error stop "FAIL: vector(real(4)) same_type_as vector(integer(4))"

  ! integer vs unsigned
  if (same_type_as(xi, xu)) &
      error stop "FAIL: vector(integer(4)) same_type_as vector(unsigned(4))"
  if (same_type_as(xu, xi)) &
      error stop "FAIL: vector(unsigned(4)) same_type_as vector(integer(4))"

  ! unsigned vs real
  if (same_type_as(xu, xr)) &
      error stop "FAIL: vector(unsigned(4)) same_type_as vector(real(4))"
  if (same_type_as(xr, xu)) &
      error stop "FAIL: vector(real(4)) same_type_as vector(unsigned(4))"

  ! same-type checks must still pass
  if (.not. same_type_as(xi, xi)) error stop "FAIL: xi same_type_as xi"
  if (.not. same_type_as(xu, xu)) error stop "FAIL: xu same_type_as xu"
  if (.not. same_type_as(xr, xr)) error stop "FAIL: xr same_type_as xr"
end subroutine

call test_allocate_source()
call test_allocate_mold()
call test_assign()
call test_cross_type()
print *, "PASS"
end

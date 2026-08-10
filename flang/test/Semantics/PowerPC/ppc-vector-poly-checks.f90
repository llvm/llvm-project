! RUN: %python %S/../test_errors.py %s %flang_fc1
! REQUIRES: target=powerpc{{.*}}

! Test that vector types are rejected in polymorphic contexts:
!   1. ALLOCATE SOURCE=/MOLD= must not have a vector type expression
!   2. RHS of intrinsic assignment must not be a vector type when LHS is polymorphic
!      (uses Fortran standard auto-allocation, no prior ALLOCATE)
!   3. SAME_TYPE_AS / EXTENDS_TYPE_OF must not receive a vector type argument
!      (enforced via IsExtensibleType returning false for vector types)

subroutine test_allocate_source()
  vector(integer(4)) :: vi(2)
  class(*), allocatable :: x(:)
  !ERROR: SOURCE or MOLD expression must not be a vector type 'vector(integer(4))'
  allocate(x(2), source=vi)
end subroutine

subroutine test_allocate_mold()
  vector(real(4)) :: vr(2)
  class(*), allocatable :: x(:)
  !ERROR: SOURCE or MOLD expression must not be a vector type 'vector(real(4))'
  allocate(x(2), mold=vr)
end subroutine

subroutine test_poly_assign_vector_rhs()
  vector(real(4)) :: vr(2)
  class(*), allocatable :: x(:)
  ! x is unallocated; auto-allocation from RHS type must be rejected
  !ERROR: Vector type 'vector(real(4))' may not be used as the right-hand side of a polymorphic intrinsic assignment
  x = vr
end subroutine

subroutine test_same_type_as()
  vector(integer(4)) :: vi(2)
  logical :: res
  !ERROR: Actual argument for 'a=' has type 'vector(integer(4))', but was expected to be an extensible or unlimited polymorphic type
  res = same_type_as(vi, vi)
end subroutine

subroutine test_extends_type_of()
  vector(real(4)) :: vr(2)
  logical :: res
  !ERROR: Actual argument for 'a=' has type 'vector(real(4))', but was expected to be an extensible or unlimited polymorphic type
  res = extends_type_of(vr, vr)
end subroutine

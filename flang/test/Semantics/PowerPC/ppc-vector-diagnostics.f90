! RUN: %python %S/../test_errors.py %s %flang_fc1
! REQUIRES: target=powerpc{{.*}}

subroutine test_vector_add()
  vector(integer(4)) :: v1, v2
  !ERROR: Operands of + must be numeric; have vector(integer(4)) and vector(integer(4))
  v1 = v1 + v2
end subroutine

subroutine test_vector_assignment()
  vector(integer(4)) :: v1
  vector(real(4)) :: v2
  !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types vector(integer(4)) and vector(real(4))
  v1 = v2
end subroutine

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

! Test scalar POINTER and ALLOCATABLE vector type declarations (not supported)
subroutine test_scalar_pointer()
  !ERROR: Pointer to vector(integer(4)) type is not supported
  vector(integer(4)), pointer :: p
end subroutine

subroutine test_scalar_allocatable()
  !ERROR: Allocatable entity of vector(integer(4)) type is not supported
  vector(integer(4)), allocatable :: v
end subroutine

! SAME_TYPE_AS with one unlimited polymorphic arg and one vector arg
subroutine test_same_type_as_mixed(x)
  class(*), intent(in) :: x
  vector(integer(4)) :: vi(2)
  logical :: res
  !ERROR: Actual argument for 'b=' has type 'vector(integer(4))', but was expected to be an extensible or unlimited polymorphic type
  res = same_type_as(x, vi)
end subroutine

! UDTI: dtv argument must not be a vector type
module test_udti_vector_read_mod
  interface read(formatted)
    module procedure rf
  end interface
contains
  subroutine rf(dtv, unit, iotype, vlist, iostat, iomsg)
    !ERROR: Dummy argument 'dtv' of a defined input/output procedure must not be a vector type
    vector(integer(4)), intent(inout) :: dtv
    integer, intent(in) :: unit
    character(*), intent(in) :: iotype
    integer, intent(in) :: vlist(:)
    integer, intent(out) :: iostat
    character(*), intent(inout) :: iomsg
  end subroutine
end module

module test_udti_vector_write_mod
  interface write(formatted)
    module procedure wf
  end interface
contains
  subroutine wf(dtv, unit, iotype, vlist, iostat, iomsg)
    !ERROR: Dummy argument 'dtv' of a defined input/output procedure must not be a vector type
    vector(integer(4)), intent(in) :: dtv
    integer, intent(in) :: unit
    character(*), intent(in) :: iotype
    integer, intent(in) :: vlist(:)
    integer, intent(out) :: iostat
    character(*), intent(inout) :: iomsg
  end subroutine
end module

! CLASSOF with a vector type data-ref must be rejected (vector types are not extensible)
subroutine test_classof_vector(c, vi)
  vector(integer(4)), intent(in) :: vi(2)
  !ERROR: CLASSOF requires a data-ref of extensible type
  classof(vi) :: c
end subroutine

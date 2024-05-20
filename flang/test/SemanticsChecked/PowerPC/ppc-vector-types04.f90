! RUN: %python %S/../test_errors.py %s %flang_fc1
! REQUIRES: target=powerpc{{.*}}

subroutine vec_type_test(arg1, arg2, arg3, arg4)
!ERROR: Assumed-shape entity of vector(real(4)) type is not supported
  vector(real) :: arg1(:)
!ERROR: Assumed Rank entity of vector(unsigned(4)) type is not supported
  vector(unsigned) :: arg2(..)
!ERROR: Deferred-shape entity of vector(integer(4)) type is not supported
  vector(integer), allocatable :: arg3(:)
!ERROR: Deferred-shape entity of vector(integer(4)) type is not supported
  vector(integer), pointer :: arg4(:)
!ERROR: Deferred-shape entity of vector(integer(4)) type is not supported
  vector(integer), allocatable :: var1(:)
!ERROR: Deferred-shape entity of vector(integer(4)) type is not supported
  vector(integer), pointer :: var2(:)
end subroutine vec_type_test

subroutine vec_pair_type_test(arg1, arg2, arg3, arg4)
!ERROR: Assumed-shape entity of __vector_pair type is not supported
  __vector_pair :: arg1(:)
!ERROR: Assumed Rank entity of __vector_pair type is not supported
  __vector_pair :: arg2(..)
!ERROR: Deferred-shape entity of __vector_pair type is not supported
  __vector_pair, allocatable :: arg3(:)
!ERROR: Deferred-shape entity of __vector_pair type is not supported
  __vector_pair, pointer :: arg4(:)
!ERROR: Deferred-shape entity of __vector_pair type is not supported
  __vector_pair, allocatable :: var1(:)
!ERROR: Deferred-shape entity of __vector_pair type is not supported
  __vector_pair, pointer :: var2(:)
end subroutine vec_pair_type_test

subroutine vec_quad_type_test(arg1, arg2, arg3, arg4)
!ERROR: Assumed-shape entity of __vector_quad type is not supported
  __vector_quad :: arg1(:)
!ERROR: Assumed Rank entity of __vector_quad type is not supported
  __vector_quad :: arg2(..)
!ERROR: Deferred-shape entity of __vector_quad type is not supported
  __vector_quad, allocatable :: arg3(:)
!ERROR: Deferred-shape entity of __vector_quad type is not supported
  __vector_quad, pointer :: arg4(:)
!ERROR: Deferred-shape entity of __vector_quad type is not supported
  __vector_quad, allocatable :: var1(:)
!ERROR: Deferred-shape entity of __vector_quad type is not supported
  __vector_quad, pointer :: var2(:)
end subroutine vec_quad_type_test

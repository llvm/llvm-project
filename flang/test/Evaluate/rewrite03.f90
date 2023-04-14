! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
! Tests rewrite of IS_CONTIGUOUS with TYPE(*) arguments.

subroutine test_is_contiguous(assumed_size, assumed_shape, &
      & assumed_shape_contiguous, assumed_rank, assumed_rank_contiguous)
  type(*) :: assumed_size(*), assumed_shape(:), assumed_shape_contiguous(:)
  type(*) :: assumed_rank(..), assumed_rank_contiguous(..)
  contiguous :: assumed_shape_contiguous, assumed_rank_contiguous
! CHECK: PRINT *, .true._4
  print *, is_contiguous(assumed_size)
! CHECK: PRINT *, .true._4
  print *, is_contiguous(assumed_shape_contiguous)
! CHECK: PRINT *, .true._4
  print *, is_contiguous(assumed_rank_contiguous)
! CHECK: PRINT *, is_contiguous(assumed_shape)
  print *, is_contiguous(assumed_shape)
! CHECK: PRINT *, is_contiguous(assumed_rank)
  print *, is_contiguous(assumed_rank)
end subroutine

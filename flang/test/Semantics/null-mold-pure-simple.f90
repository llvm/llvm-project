! RUN: %flang_fc1 -fsyntax-only %s

! Verify that NULL(MOLD=) is PURE and can be used in a PURE procedure
module null_mold_pure_simple
  implicit none
contains
  pure subroutine reset(p)
    real, pointer, intent(inout) :: p
    p => null(p)
  end subroutine reset
end module null_mold_pure_simple

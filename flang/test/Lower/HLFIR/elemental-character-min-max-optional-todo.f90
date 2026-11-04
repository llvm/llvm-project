! Test that elemental character MIN/MAX with dynamically optional arguments
! correctly hits the TODO (not yet implemented) diagnostic.
! RUN: %not_todo_cmd bbc -emit-hlfir -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: CHARACTER MIN and MAX with dynamically optional arguments
subroutine test_elemental_char_min_assumed_optional(a, b, c, res)
  character(*), intent(in) :: a(:), b(:)
  character(*), intent(in), optional :: c(:)
  character(*), intent(out) :: res(:)
  res = min(a, b, c)
end subroutine

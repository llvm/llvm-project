! RUN: %flang_fc1 -emit-llvm %s -o -

! Test that select case with pointer assignment compiles correctly.
! This requires block signature conversion in SelectCaseOpConversion.
subroutine test(l)
  integer :: l
  integer, pointer :: p(:)
  integer, target :: a1(2), a2(2)

  select case (l)
    case (1)
      p => a1
    case (2)
      p => a2
  end select

  p = 0
end subroutine

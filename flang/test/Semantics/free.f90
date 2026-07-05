! RUN: %python %S/test_errors.py %s %flang_fc1 -Werror

! Accept free of cray pointer without warning
subroutine free_cptr()
  integer :: x
  pointer(ptr_x, x)
  call free(ptr_x)
end subroutine

subroutine free_i8()
  integer(kind=1) :: x
  ! WARNING: FREE should only be used with Cray pointers
  call free(x)
end subroutine


subroutine free_i16()
  integer(kind=2) :: x
  ! WARNING: FREE should only be used with Cray pointers
  call free(x)
end subroutine

subroutine free_i32()
  integer(kind=4) :: x
  ! WARNING: FREE should only be used with Cray pointers
  call free(x)
end subroutine

subroutine free_i64()
  integer(kind=8) :: x
  ! WARNING: FREE should only be used with Cray pointers
  call free(x)
end subroutine

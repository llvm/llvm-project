module message
  type :: msg_type
    integer, pointer :: arr(:)
  end type msg_type
  contains
  subroutine test(msg, res)
    type(msg_type) :: msg
    integer, intent(out) :: res
    res = is_contiguous(msg%arr)
  end subroutine
end module

program prog
  use message

  integer, parameter :: num = 1
  integer rslts(num), expect(num)
  data expect / true /

  type(msg_type) :: my_msg
  integer :: res_test
  call test(my_msg, res_test)

  ! test that summing elements of an array where the array lives in another
  ! module produces the correct result.
  rslts(1) = res_test
  call check(rslts, expect, num)

  print *, "PASSED" ! Check that this compiles without error.
end program prog

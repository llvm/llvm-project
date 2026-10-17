integer function lib_add_two(n)
  !dir$ inlinealways lib_add_two
  integer :: n
  lib_add_two = n + 2
end function

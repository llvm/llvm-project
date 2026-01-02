! RUN: %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck --allow-empty %s
! CHECK-NOT: error:
! Test that zero-length character substrings with lbound>ubounds indices
! do not produce spurious errors when the character array is zero length.

integer function init(i)
  integer i
  init=i
end

program flang_t7052
  character*(*), parameter :: param_char = ""
  character*(0)            :: zero_len_char

  if ( param_char(init(5):init(3)) > zero_len_char(1:-2) ) then
    print *,"Test failed"
  endif

  if ( param_char(init(5):init(3)) > zero_len_char(10:2) ) then
    print *,"Test failed"
  endif

  if ( param_char(init(5):init(3)) > zero_len_char(init(10):2) ) then
    print *,"Test failed"
  endif

  if ( param_char(init(5):init(3)) > zero_len_char(init(10):-2) ) then
    print *,"Test failed"
  endif
end program

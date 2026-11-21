! REQUIRES: system-linux
! UNSUPPORTED: asan, hwasan, msan, tsan
! Expanding a scalar CHARACTER value into a large array component must not
! create a separate string for each element.
! RUN: ulimit -v 1048576
! RUN: %flang_fc1 -fsyntax-only %s
subroutine zero_length
  type t
    character(0) c(2000000000)
  end type
  type(t) :: x = t('')
end

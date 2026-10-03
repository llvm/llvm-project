! RUN: %python %S/test_folding.py %s %flang_fc1 -fdefault-integer-8

! Verify that folding OUT_OF_RANGE works when the default integer and
! logical kinds are promoted to kind 8.

module test_out_of_range_default_integer_8
    integer, parameter :: i1 = selected_int_kind(2)
    integer, parameter :: wp = selected_real_kind(6)
  
    logical, parameter :: expected(*) = &
        [.false., .true., .false., .true.]
  
    logical, parameter :: result(*) = out_of_range( &
        [127.0_wp, 128.0_wp, -128.0_wp, -129.0_wp], 0_i1)
  
    logical, parameter :: test_result = all(result .eqv. expected)
  end module

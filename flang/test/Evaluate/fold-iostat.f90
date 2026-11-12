! RUN: %python %S/test_folding.py %s %flang_fc1
module m
  use iso_fortran_env
  logical, parameter :: test_end1 = is_iostat_end(iostat_end)
  logical, parameter :: test_end2 = .not. is_iostat_end(iostat_eor)
  logical, parameter :: test_eor1 = is_iostat_eor(iostat_eor)
  logical, parameter :: test_eor2 = .not. is_iostat_eor(iostat_end)
  logical, parameter :: test_arr1 = &
    all(is_iostat_end([iostat_end, iostat_eor]) .eqv. [.true., .false.])
  logical, parameter :: test_arr2 = &
    all(is_iostat_eor([iostat_end, iostat_eor]) .eqv. [.false., .true.])
end

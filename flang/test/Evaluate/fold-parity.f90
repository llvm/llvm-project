! RUN: %python %S/test_folding.py %s %flang_fc1

! Test fold parity intrinsic.

module paritytest
  logical, parameter :: test_1t = parity((/ .true. /))
  logical, parameter :: test_1f = .not. parity((/ .false. /))

  logical, parameter :: test_e1 = .not. parity((/ .true., .true. /))
  logical, parameter :: test_o1 = parity((/ .true., .true., .true. /))
  logical, parameter :: test_o12 = parity((/ .true., .true., .true., .false. /))

  logical, parameter, dimension(2, 3) :: a32 = reshape((/&
       .true., .true., .false., &
       .true., .true., .true. &
       /), shape(a32), order=(/2, 1/))

  logical, parameter, dimension(2, 3) :: a32t = reshape((/&
       .true., .true., .true., &
       .true., .true., .true. &
       /), shape(a32t))

  logical, parameter, dimension(2, 3) :: a32f = reshape((/&
       .false., .false., .false., &
       .false., .false., .false. &
       /), shape(a32f))

  logical, parameter :: test_a32 = parity(a32)
  logical, parameter :: test_a32t = .not. parity(a32t)
  logical, parameter :: test_a32f = .not. parity(a32f)

  logical, parameter :: test_a321 = &
       all(parity(a32, 1) .EQV. (/ .false., .false., .true. /))

  logical, parameter :: test_a322 = &
       all(parity(a32, 2) .EQV. (/ .false., .true. /))

end module paritytest

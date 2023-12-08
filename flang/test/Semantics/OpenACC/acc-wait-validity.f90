! RUN: %python %S/../test_errors.py %s %flang -fopenacc

! Check OpenACC clause validity for the following construct and directive:
!   2.16.13 Wait

program openacc_wait_validity

  implicit none

  logical :: ifCondition = .TRUE.

  !$acc wait

  !$acc wait async

  !$acc wait(1)
  !$acc wait(1, 2)

  !$acc wait(queues: 1)
  !$acc wait(queues: 1, 2)

  !$acc wait(devnum: 1: 3)
  !$acc wait(devnum: 1: 3, 4)

  !$acc wait(devnum: 1: queues: 3)
  !$acc wait(devnum: 1: queues: 3, 4)

  !$acc wait(1) if(.true.)

  !ERROR: At most one IF clause can appear on the WAIT directive
  !$acc wait(1) if(.true.) if(.false.)

  !$acc wait(1) if(.true.) async

  !$acc wait(1) if(ifCondition) async

  !$acc wait(1) if(.true.) async(1)

  !ERROR: At most one ASYNC clause can appear on the WAIT directive
  !$acc wait(1) if(.true.) async(1) async

end program openacc_wait_validity

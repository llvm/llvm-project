! RUN: %python %S/../test_errors.py %s %flang -fopenacc

! Check OpenACC clause validity for the following construct and directive:
!   2.14.4 Update

program openacc_update_validity

  implicit none

  type atype
    real(8), dimension(10) :: arr
  end type atype

  integer :: i
  integer, parameter :: N = 256
  integer, dimension(N) :: c
  integer :: async1
  integer :: wait1, wait2
  real(8), dimension(N, N) :: aa, bb, cc
  logical :: ifCondition = .TRUE.
  type(atype) :: t
  type(atype), dimension(10) :: ta
  real(8), dimension(N) :: a, f, g, h

  !ERROR: At least one of DEVICE, HOST, SELF clause must appear on the UPDATE directive
  !$acc update

  !$acc update device(t%arr(:))

  !$acc update device(ta(i)%arr(:))

  !$acc update self(a, f) host(g) device(h)

  !$acc update host(aa) async(1)

  !$acc update device(bb) async(async1)

  !ERROR: At most one ASYNC clause can appear on the UPDATE directive
  !$acc update host(aa, bb) async(1) async(2)

  !$acc update self(bb, cc(:)) wait(1)

  !ERROR: SELF clause on the UPDATE directive must have a var-list
  !$acc update self

  !$acc update device(aa, bb, cc) wait(wait1)

  !$acc update host(aa) host(bb) device(cc) wait(1,2)

  !$acc update device(aa, cc) wait(wait1, wait2)

  !$acc update device(aa) device_type(*) async

  !$acc update host(bb) device_type(*) wait

  !$acc update self(cc) device_type(1,2) async device_type(3) wait

  !ERROR: At most one IF clause can appear on the UPDATE directive
  !$acc update device(aa) if(.true.) if(ifCondition)

  !ERROR: At most one IF_PRESENT clause can appear on the UPDATE directive
  !$acc update device(bb) if_present if_present

  !ERROR: Clause IF is not allowed after clause DEVICE_TYPE on the UPDATE directive
  !$acc update device(i) device_type(*) if(.TRUE.)

end program openacc_update_validity

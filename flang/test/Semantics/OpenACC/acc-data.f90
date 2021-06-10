! RUN: %S/../test_errors.sh %s %t %flang -fopenacc
! REQUIRES: shell

! Check OpenACC clause validity for the following construct and directive:
!   2.6.5 Data
!   2.14.6 Enter Data
!   2.14.7 Exit Data

program openacc_data_validity

  implicit none

  type atype
    real(8), dimension(10) :: arr
    real(8) :: s
  end type atype

  integer :: i, j, b, gang_size, vector_size, worker_size
  integer, parameter :: N = 256
  integer, dimension(N) :: c
  logical, dimension(N) :: d, e
  integer :: async1
  integer :: wait1, wait2
  real :: reduction_r
  logical :: reduction_l
  real(8), dimension(N, N) :: aa, bb, cc
  real(8), dimension(:), allocatable :: dd
  real(8), pointer :: p
  logical :: ifCondition = .TRUE.
  type(atype) :: t
  type(atype), dimension(10) :: ta

  real(8), dimension(N) :: a, f, g, h

  !ERROR: At least one of ATTACH, COPYIN, CREATE clause must appear on the ENTER DATA directive
  !$acc enter data

  !ERROR: Modifier is not allowed for the COPYIN clause on the ENTER DATA directive
  !$acc enter data copyin(zero: i)

  !ERROR: Only the ZERO modifier is allowed for the CREATE clause on the ENTER DATA directive
  !$acc enter data create(readonly: i)

  !ERROR: COPYOUT clause is not allowed on the ENTER DATA directive
  !$acc enter data copyin(i) copyout(i)

  !$acc enter data create(aa) if(.TRUE.)

  !ERROR: At most one IF clause can appear on the ENTER DATA directive
  !$acc enter data create(aa) if(.TRUE.) if(ifCondition)

  !$acc enter data create(aa) if(ifCondition)

  !$acc enter data create(aa) async

  !ERROR: At most one ASYNC clause can appear on the ENTER DATA directive
  !$acc enter data create(aa) async async

  !$acc enter data create(aa) async(async1)

  !$acc enter data create(aa) async(1)

  !$acc enter data create(aa) wait(1)

  !$acc enter data create(aa) wait(wait1)

  !$acc enter data create(aa) wait(wait1, wait2)

  !$acc enter data create(aa) wait(wait1) wait(wait2)

  !ERROR: Argument `bb` on the ATTACH clause must be a variable or array with the POINTER or ALLOCATABLE attribute
  !$acc enter data attach(bb)

  !ERROR: At least one of COPYOUT, DELETE, DETACH clause must appear on the EXIT DATA directive
  !$acc exit data

  !ERROR: Modifier is not allowed for the COPYOUT clause on the EXIT DATA directive
  !$acc exit data copyout(zero: i)

  !$acc exit data delete(aa)

  !$acc exit data delete(aa) finalize

  !ERROR: At most one FINALIZE clause can appear on the EXIT DATA directive
  !$acc exit data delete(aa) finalize finalize

  !ERROR: Argument `cc` on the DETACH clause must be a variable or array with the POINTER or ALLOCATABLE attribute
  !$acc exit data detach(cc)

  !ERROR: Argument on the DETACH clause must be a variable or array with the POINTER or ALLOCATABLE attribute
  !$acc exit data detach(/i/)

  !$acc exit data copyout(bb)

  !$acc exit data delete(aa) if(.TRUE.)

  !$acc exit data delete(aa) if(ifCondition)

  !ERROR: At most one IF clause can appear on the EXIT DATA directive
  !$acc exit data delete(aa) if(ifCondition) if(.TRUE.)

  !$acc exit data delete(aa) async

  !ERROR: At most one ASYNC clause can appear on the EXIT DATA directive
  !$acc exit data delete(aa) async async

  !$acc exit data delete(aa) async(async1)

  !$acc exit data delete(aa) async(1)

  !$acc exit data delete(aa) wait(1)

  !$acc exit data delete(aa) wait(wait1)

  !$acc exit data delete(aa) wait(wait1, wait2)

  !$acc exit data delete(aa) wait(wait1) wait(wait2)

  !ERROR: Only the ZERO modifier is allowed for the COPYOUT clause on the DATA directive
  !$acc data copyout(readonly: i)
  !$acc end data

  !ERROR: At most one IF clause can appear on the DATA directive
  !$acc data copy(i) if(.true.) if(.true.)
  !$acc end data

  !ERROR: At least one of COPYOUT, DELETE, DETACH clause must appear on the EXIT DATA directive
  !$acc exit data

  !ERROR: At least one of ATTACH, COPY, COPYIN, COPYOUT, CREATE, DEFAULT, DEVICEPTR, NO_CREATE, PRESENT clause must appear on the DATA directive
  !$acc data
  !$acc end data

  !$acc data copy(aa) if(.true.)
  !$acc end data

  !$acc data copy(aa) if(ifCondition)
  !$acc end data

  !$acc data copy(aa, bb, cc)
  !$acc end data

  !$acc data copyin(aa) copyin(readonly: bb) copyout(cc)
  !$acc end data

  !$acc data copyin(readonly: aa, bb) copyout(zero: cc)
  !$acc end data

  !$acc data create(aa, bb(:,:)) create(zero: cc(:,:))
  !$acc end data

  !$acc data no_create(aa) present(bb, cc)
  !$acc end data

  !$acc data deviceptr(aa) attach(dd, p)
  !$acc end data

  !$acc data copy(aa, bb) default(none)
  !$acc end data

  !$acc data copy(aa, bb) default(present)
  !$acc end data

  !ERROR: At most one DEFAULT clause can appear on the DATA directive
  !$acc data copy(aa, bb) default(none) default(present)
  !$acc end data

  !ERROR: At most one IF clause can appear on the DATA directive
  !$acc data copy(aa) if(.true.) if(ifCondition)
  !$acc end data

  !$acc data copyin(i)
  !ERROR: Unmatched PARALLEL directive
  !$acc end parallel

end program openacc_data_validity

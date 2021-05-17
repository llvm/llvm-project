! RUN: %S/../test_errors.sh %s %t %flang -fopenacc

! Check OpenACC clause validity for the following construct and directive:
!   2.8 host_data

program openacc_host_data_validity

  implicit none

  integer, parameter :: N = 256
  real(8), dimension(N, N) :: aa, bb
  logical :: ifCondition = .TRUE.

  !ERROR: At least one of USE_DEVICE clause must appear on the HOST_DATA directive
  !$acc host_data
  !$acc end host_data

  !$acc host_data use_device(aa)
  !$acc end host_data

  !$acc host_data use_device(aa) if(.true.)
  !$acc end host_data

  !$acc host_data use_device(aa) if(ifCondition)
  !$acc end host_data

  !$acc host_data use_device(aa, bb) if_present
  !$acc end host_data

  !ERROR: At most one IF_PRESENT clause can appear on the HOST_DATA directive
  !$acc host_data use_device(aa, bb) if_present if_present
  !$acc end host_data

  !$acc host_data use_device(aa, bb) if(.true.) if_present
  !$acc end host_data

  !ERROR: At most one IF clause can appear on the HOST_DATA directive
  !$acc host_data use_device(aa, bb) if(.true.) if(ifCondition)
  !$acc end host_data

end program openacc_host_data_validity

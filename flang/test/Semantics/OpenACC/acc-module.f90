! RUN: %python %S/../test_modfile.py %s %flang_fc1 -fopenacc

module acc_mod
  real :: data_create(100)
  !$acc declare create(data_create)

  real :: data_copyin(10)
  !$acc declare copyin(data_copyin)

  real :: data_device_resident(20)
  !$acc declare device_resident(data_device_resident)

  integer :: data_link(50)
  !$acc declare link(data_link)
end module

!Expect: acc_mod.mod
! module acc_mod
! real(4)::data_create(1_8:100_8)
! !$acc declare create(data_create)
! real(4)::data_copyin(1_8:10_8)
! !$acc declare copyin(data_copyin)
! real(4)::data_device_resident(1_8:20_8)
! !$acc declare device_resident(data_device_resident)
! integer(4)::data_link(1_8:50_8)
! !$acc declare link(data_link)
! end

! RUN: %python %S/../test_modfile.py %s %flang_fc1 -fopenacc

module acc_mod
  real :: data_create(100)
  !$acc declare create(data_create)

  real :: data_copyin(10)
  !$acc declare copyin(data_copyin)

  real :: data_copyinro(10)
  !$acc declare copyin(readonly: data_copyinro)

  real :: data_device_resident(20)
  !$acc declare device_resident(data_device_resident)

  integer :: data_link(50)
  !$acc declare link(data_link)

  !$acc routine(sub10) seq

contains
  subroutine sub1()
    !$acc routine
  end subroutine

  subroutine sub2()
    !$acc routine seq
  end subroutine

  subroutine sub3()
    !$acc routine gang
  end subroutine

  subroutine sub4()
    !$acc routine vector
  end subroutine

  subroutine sub5()
    !$acc routine worker
  end subroutine

  subroutine sub6()
    !$acc routine gang(dim:2)
  end subroutine

  subroutine sub7()
    !$acc routine bind("sub7_")
  end subroutine

  subroutine sub8()
    !$acc routine bind(sub7)
  end subroutine

  subroutine sub9()
    !$acc routine vector
    !$acc routine seq bind(sub7)
    !$acc routine gang bind(sub8)
  end subroutine

  subroutine sub10()
  end subroutine

end module

!Expect: acc_mod.mod
! module acc_mod
! real(4)::data_create(1_8:100_8)
! !$acc declare create(data_create)
! real(4)::data_copyin(1_8:10_8)
! !$acc declare copyin(data_copyin)
! real(4)::data_copyinro(1_8:10_8)
! !$acc declare copyin(readonly: data_copyinro)
! real(4)::data_device_resident(1_8:20_8)
! !$acc declare device_resident(data_device_resident)
! integer(4)::data_link(1_8:50_8)
! !$acc declare link(data_link)
! contains
! subroutine sub1()
! !$acc routine
! end
! subroutine sub2()
! !$acc routine seq
! end
! subroutine sub3()
! !$acc routine gang
! end
! subroutine sub4()
! !$acc routine vector
! end
! subroutine sub5()
! !$acc routine worker
! end
! subroutine sub6()
! !$acc routine gang(dim:2)
! end
! subroutine sub7()
! !$acc routine bind("sub7_")
! end
! subroutine sub8()
! !$acc routine bind(sub7)
! end
! subroutine sub9()
! !$acc routine vector
! !$acc routine seq bind(sub7)
! !$acc routine gang bind(sub8)
! end
! subroutine sub10()
! !$acc routine seq
! end
! end

! RUN: bbc -o - %s | FileCheck %s

module units
  integer, parameter :: preconnected_unit(3) = [0, 5, 6]
contains
  ! CHECK-LABEL: _QMunitsPis_preconnected_unit
  logical function is_preconnected_unit(u)
  ! CHECK: [[units_ssa:%[0-9]+]] = fir.address_of([[units_value:.*]]) :
  ! !fir.ref<!fir.array<3xi32>>
    integer :: u
    integer :: i
    is_preconnected_unit = .true.
    do i = lbound(preconnected_unit,1), ubound(preconnected_unit,1)
      ! CHECK: fir.coordinate_of [[units_ssa]]
      if (preconnected_unit(i) == u) return
    end do
    is_preconnected_unit = .false.
  end function
end module units

! CHECK-LABEL: _QPcheck_units
subroutine check_units
  use units
  do i=-1,8
    if (is_preconnected_unit(i)) print*, i
  enddo
end

! CHECK-LABEL: _QQmain
program prog
  call check_units
end

! CHECK: fir.global linkonce [[units_value]] constant : !fir.array<3xi32>

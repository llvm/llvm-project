! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

module units
  integer, parameter :: preconnected_unit(3) = [0, 5, 6]
contains
  ! CHECK-LABEL: func.func @_QMunitsPis_preconnected_unit
  logical function is_preconnected_unit(u)
  ! CHECK: [[units_ssa:%[0-9]+]] = fir.address_of(@_QMunitsECpreconnected_unit) : !fir.ref<!fir.array<3xi32>>
  ! CHECK: [[units_decl:%[0-9]+]]:2 = hlfir.declare [[units_ssa]]
    integer :: u
    integer :: i
    is_preconnected_unit = .true.
    do i = lbound(preconnected_unit,1), ubound(preconnected_unit,1)
      ! CHECK: hlfir.designate [[units_decl]]#0
      if (preconnected_unit(i) == u) return
    end do
    is_preconnected_unit = .false.
  end function
end module units

! CHECK-LABEL: func.func @_QPcheck_units
subroutine check_units
  use units
  do i=-1,8
    if (is_preconnected_unit(i)) print*, i
  enddo
end

! CHECK-LABEL: func.func @_QPzero
subroutine zero
  complex, parameter :: a(0) = [(((k,k=1,10),j=-2,2,-1),i=2,-2,-2)]
  complex, parameter :: b(0) = [(7,i=3,-3)]
  ! CHECK: fir.address_of(@_QQro.0xz4.null.0) : !fir.ref<!fir.array<0xcomplex<f32>>>
  ! CHECK: hlfir.declare {{.*}} {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.0xz4.null.0"}
  print*, '>', a, '<'
  print*, '>', b, '<'
end

! CHECK-LABEL: func.func @_QQmain
program prog
  call check_units
  call zero
end

! CHECK: fir.global internal @_QFzeroECa constant : !fir.array<0xcomplex<f32>>
! CHECK:   %0 = fir.undefined !fir.array<0xcomplex<f32>>
! CHECK:   fir.has_value %0 : !fir.array<0xcomplex<f32>>

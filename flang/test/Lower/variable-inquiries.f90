! Test property inquiries on variables
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

module inquired
  real(8), allocatable :: a(:)
end module

! CHECK-LABEL: @_QPissue844()
subroutine issue844()
  use inquired
  ! Verify that evaluate::DescriptorInquiry are made using the symbol mapped
  ! in lowering (the use associated one, and not directly the ultimate
  ! symbol).

  ! CHECK: %[[a:.*]] = fir.address_of(@_QMinquiredEa) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
  ! CHECK: %[[adecl:.*]]:2 = hlfir.declare %[[a]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMinquiredEa"}
  ! CHECK: %[[box_load:.*]] = fir.load %[[adecl]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
  ! CHECK: %[[dim:.*]]:3 = fir.box_dims %[[box_load]], %c0{{.*}} : (!fir.box<!fir.heap<!fir.array<?xf64>>>, index) -> (index, index, index)
  ! CHECK: %[[cast:.*]] = fir.convert %[[dim]]#1 : (index) -> i64
  ! CHECK: fir.call @_FortranAioOutputInteger64(%{{.*}}, %[[cast]]) {{.*}}: (!fir.ref<i8>, i64) -> i1
  print *, size(a, kind=8)
end subroutine

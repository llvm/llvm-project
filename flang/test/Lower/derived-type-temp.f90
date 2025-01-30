! Test lowering of derived type temporary creation and init
! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s

program derived_temp_init
  type t1
    integer, allocatable :: i
  end type
  type t2
    type(t1) :: c
  end type
  type(t1) :: x
  type(t2) :: y
  y = t2(x)
end

! CHECK: %[[x:.*]] =  fir.alloca !fir.type<_QFTt1{i:!fir.box<!fir.heap<i32>>}> {bindc_name = "x", uniq_name = "_QFEx"}
! CHECK: %[[ADDR:.*]] = fir.address_of(@_QQ_QFTt1.DerivedInit) : !fir.ref<!fir.type<_QFTt1{i:!fir.box<!fir.heap<i32>>}>>
! CHECK: %[[LOAD:.*]] = fir.load %[[ADDR]] : !fir.ref<!fir.type<_QFTt1{i:!fir.box<!fir.heap<i32>>}>>
! CHECK: fir.store %[[LOAD]] to %[[x]] : !fir.ref<!fir.type<_QFTt1{i:!fir.box<!fir.heap<i32>>}>>

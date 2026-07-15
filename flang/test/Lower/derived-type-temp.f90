! Test lowering of derived type temporary creation and init
! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

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

! CHECK: %[[ALLOC:.*]] =  fir.alloca !fir.type<_QFTt1{i:!fir.box<!fir.heap<i32>>}> {bindc_name = "x", uniq_name = "_QFEx"}
! CHECK: %[[x:.*]]:2 = hlfir.declare %[[ALLOC]] {{.*}}
! CHECK: %[[COOR:.*]] = fir.coordinate_of %[[x]]#0, i : (!fir.ref<!fir.type<_QFTt1{i:!fir.box<!fir.heap<i32>>}>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK: %[[ZERO:.*]] = fir.zero_bits !fir.heap<i32>
! CHECK: %[[EMBOX:.*]] = fir.embox %[[ZERO]] : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
! CHECK: fir.store %[[EMBOX]] to %[[COOR]] : !fir.ref<!fir.box<!fir.heap<i32>>>

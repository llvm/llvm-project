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

! CHECK: %[[temp:.*]] = fir.alloca !fir.type<_QFTt1{i:!fir.box<!fir.heap<i32>>}> {bindc_name = "x", uniq_name = "_QFEx"}
! CHECK: %[[box:.*]] = fir.embox %[[temp]] : (!fir.ref<!fir.type<_QFTt1{i:!fir.box<!fir.heap<i32>>}>>) -> !fir.box<!fir.type<_QFTt1{i:!fir.box<!fir.heap<i32>>}>>
! CHECK: %[[box_none:.*]] = fir.convert %[[box]] : (!fir.box<!fir.type<_QFTt1{i:!fir.box<!fir.heap<i32>>}>>) -> !fir.box<none>
! CHECK: fir.call @_FortranAInitialize(%[[box_none]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.box<none>, !fir.ref<i8>, i32) -> ()

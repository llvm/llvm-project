! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

subroutine trans_test(store, word)
    ! CHECK-LABEL: func @_QPtrans_test(
    ! CHECK-SAME:                      %[[VAL_0:.*]]: !fir.ref<i32>{{.*}}, %[[VAL_1:.*]]: !fir.ref<f32>{{.*}}) {
    ! CHECK-DAG:     %[[RESULT_BOX:.*]] = fir.alloca !fir.box<!fir.heap<i32>>
    ! CHECK-DAG:     %[[store:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}{uniq_name = "_QFtrans_testEstore"}
    ! CHECK-DAG:     %[[word:.*]]:2 = hlfir.declare %[[VAL_1]] {{.*}}{uniq_name = "_QFtrans_testEword"}
    ! CHECK:         %[[VAL_3:.*]] = fir.embox %[[word]]#0 : (!fir.ref<f32>) -> !fir.box<f32>
    ! CHECK:         %[[VAL_4:.*]] = fir.embox %[[store]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
    ! CHECK:         fir.call @_FortranATransfer({{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
    ! CHECK:         %[[LOADED:.*]] = fir.load %[[RESULT_BOX]] : !fir.ref<!fir.box<!fir.heap<i32>>>
    ! CHECK:         %[[ADDR:.*]] = fir.box_addr %[[LOADED]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
    ! CHECK:         %[[VAL:.*]] = fir.load %[[ADDR]] : !fir.heap<i32>
    ! CHECK:         fir.freemem %[[ADDR]]
    ! CHECK:         hlfir.assign %[[VAL]] to %[[store]]#0 : i32, !fir.ref<i32>
    ! CHECK:         return
    ! CHECK:       }
    integer :: store
    real :: word
    store = transfer(word, store)
  end subroutine

  ! CHECK-LABEL: func @_QPtrans_test2(
  subroutine trans_test2(store, word)
    ! CHECK-DAG:     %[[RESULT_BOX:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
    ! CHECK-DAG:     %[[storeDecl:.*]]:2 = hlfir.declare {{.*}}{uniq_name = "_QFtrans_test2Estore"}
    ! CHECK-DAG:     %[[wordDecl:.*]]:2 = hlfir.declare {{.*}}{uniq_name = "_QFtrans_test2Eword"}
    ! CHECK:         fir.call @_FortranATransferSize({{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32, i64) -> ()
    ! CHECK:         hlfir.assign {{.*}} to %[[storeDecl]]#0
    ! CHECK:         hlfir.destroy {{.*}}
    ! CHECK:         return
    integer :: store(3)
    real :: word
    store = transfer(word, store, 3)
  end subroutine

  integer function trans_test3(p)
    ! CHECK-LABEL: func @_QPtrans_test3(
    ! CHECK-SAME:                       %[[VAL_0:.*]]: !fir.ref<i32>{{.*}}) -> i32 {
    ! CHECK-DAG:     %[[pDecl:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}{uniq_name = "_QFtrans_test3Ep"}
    ! CHECK-DAG:     %[[tDecl:.*]]:2 = hlfir.declare {{.*}}{uniq_name = "_QFtrans_test3Et"}
    ! CHECK:         fir.call @_FortranATransfer({{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
    ! CHECK:         hlfir.assign {{.*}} to %[[tDecl]]#0
    ! CHECK:         %[[x_field:.*]] = hlfir.designate %[[tDecl]]#0{"x"}
    ! CHECK:         %[[x_val:.*]] = fir.load %[[x_field]] : !fir.ref<i32>
    ! CHECK:         return %{{.*}} : i32
    ! CHECK:       }
    type obj
      integer :: x
    end type
    type (obj) :: t
    integer :: p
    t = transfer(p, t)
    trans_test3 = t%x
  end function

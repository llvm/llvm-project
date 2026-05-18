! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

subroutine trans_test(store, word)
    ! CHECK-LABEL: func @_QPtrans_test(
    ! CHECK-SAME:                      %[[VAL_0:.*]]: !fir.ref<i32>{{.*}}, %[[VAL_1:.*]]: !fir.ref<f32>{{.*}}) {
    ! CHECK-DAG:     %[[store:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}{uniq_name = "_QFtrans_testEstore"}
    ! CHECK-DAG:     %[[word:.*]]:2 = hlfir.declare %[[VAL_1]] {{.*}}{uniq_name = "_QFtrans_testEword"}
    ! CHECK:         %[[LOADED:.*]] = fir.load %[[word]]#0 : !fir.ref<f32>
    ! CHECK:         %[[VAL:.*]] = arith.bitcast %[[LOADED]] : f32 to i32
    ! CHECK:         hlfir.assign %[[VAL]] to %[[store]]#0 : i32, !fir.ref<i32>
    ! CHECK-NOT:     fir.call @_FortranATransfer
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

  ! Scalar same-size transfer (f64 -> i64) is inlined as fir.load + arith.bitcast.
  subroutine trans_test_r8_to_i8(store, word)
    ! CHECK-LABEL: func @_QPtrans_test_r8_to_i8(
    ! CHECK-SAME:    %[[RES:.*]]: !fir.ref<i64>{{.*}}, %[[SRC:.*]]: !fir.ref<f64>{{.*}}) {
    ! CHECK-DAG:     %[[store:.*]]:2 = hlfir.declare %[[RES]] {{.*}}{uniq_name = "_QFtrans_test_r8_to_i8Estore"}
    ! CHECK-DAG:     %[[word:.*]]:2 = hlfir.declare %[[SRC]] {{.*}}{uniq_name = "_QFtrans_test_r8_to_i8Eword"}
    ! CHECK:         %[[LOADED:.*]] = fir.load %[[word]]#0 : !fir.ref<f64>
    ! CHECK:         %[[VAL:.*]] = arith.bitcast %[[LOADED]] : f64 to i64
    ! CHECK:         hlfir.assign %[[VAL]] to %[[store]]#0 : i64, !fir.ref<i64>
    ! CHECK-NOT:     fir.call @_FortranATransfer
    ! CHECK:         return
    ! CHECK:       }
    integer(8) :: store
    real(8) :: word
    store = transfer(word, store)
  end subroutine

  ! BIND(C) struct (c_ptr) to integer(8): same byte size, inlined via
  ! address-level reinterpret. Covers the c_devptr pattern on CUDA device code.
  subroutine trans_test_cptr_to_i8(store, src)
    ! CHECK-LABEL: func @_QPtrans_test_cptr_to_i8(
    ! CHECK:         %[[srcDecl:.*]]:2 = hlfir.declare {{.*}}{uniq_name = "_QFtrans_test_cptr_to_i8Esrc"}
    ! CHECK:         %[[storeDecl:.*]]:2 = hlfir.declare {{.*}}{uniq_name = "_QFtrans_test_cptr_to_i8Estore"}
    ! CHECK:         %[[CAST:.*]] = fir.convert %[[srcDecl]]#0 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
    ! CHECK:         %[[VAL:.*]] = fir.load %[[CAST]] : !fir.ref<i64>
    ! CHECK:         hlfir.assign %[[VAL]] to %[[storeDecl]]#0 : i64, !fir.ref<i64>
    ! CHECK-NOT:     fir.call @_FortranATransfer
    ! CHECK:         return
    ! CHECK:       }
    use iso_c_binding
    integer(8) :: store
    type(c_ptr) :: src
    store = transfer(src, store)
  end subroutine

  ! Different-size scalar transfer (i32 -> i64) falls back to runtime.
  subroutine trans_test_diff_size(store, src)
    ! CHECK-LABEL: func @_QPtrans_test_diff_size(
    ! CHECK:         fir.call @_FortranATransfer(
    ! CHECK:         return
    ! CHECK:       }
    integer(8) :: store
    integer(4) :: src
    store = transfer(src, store)
  end subroutine

  ! Array mold without SIZE: result is rank-1 array, must use runtime.
  subroutine trans_test_array_mold(src, result)
    ! CHECK-LABEL: func @_QPtrans_test_array_mold(
    ! CHECK:         fir.call @_FortranATransfer(
    ! CHECK:         return
    ! CHECK:       }
    real :: src
    integer, allocatable :: result(:)
    integer :: mold(4)
    result = transfer(src, mold)
  end subroutine

  ! Allocatable mold: must use runtime.
  subroutine trans_test_alloc_mold(src, result)
    ! CHECK-LABEL: func @_QPtrans_test_alloc_mold(
    ! CHECK:         fir.call @_FortranATransfer(
    ! CHECK:         return
    ! CHECK:       }
    real :: src
    integer, allocatable :: mold(:)
    integer, allocatable :: result(:)
    result = transfer(src, mold)
  end subroutine

  ! POINTER source: descriptor is unpacked before reaching genTransfer,
  ! so the inline optimization applies.
  subroutine trans_test_pointer_source(store, src)
    ! CHECK-LABEL: func @_QPtrans_test_pointer_source(
    ! CHECK:         fir.load {{.*}} : !fir.ref<!fir.box<!fir.ptr<f32>>>
    ! CHECK:         fir.box_addr
    ! CHECK:         %[[VAL:.*]] = fir.load {{.*}} : !fir.ptr<f32>
    ! CHECK:         arith.bitcast %[[VAL]] : f32 to i32
    ! CHECK-NOT:     fir.call @_FortranATransfer
    ! CHECK:         return
    ! CHECK:       }
    integer :: store
    real, pointer :: src
    store = transfer(src, store)
  end subroutine

  ! ALLOCATABLE source: descriptor is unpacked before reaching genTransfer,
  ! so the inline optimization applies.
  subroutine trans_test_alloc_source(store, src)
    ! CHECK-LABEL: func @_QPtrans_test_alloc_source(
    ! CHECK:         fir.load {{.*}} : !fir.ref<!fir.box<!fir.heap<f32>>>
    ! CHECK:         fir.box_addr
    ! CHECK:         %[[VAL:.*]] = fir.load {{.*}} : !fir.heap<f32>
    ! CHECK:         arith.bitcast %[[VAL]] : f32 to i32
    ! CHECK-NOT:     fir.call @_FortranATransfer
    ! CHECK:         return
    ! CHECK:       }
    integer :: store
    real, allocatable :: src
    store = transfer(src, store)
  end subroutine

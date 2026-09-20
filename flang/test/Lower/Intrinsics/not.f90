! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

subroutine not_test
    integer :: source
    integer :: destination
    ! CHECK-LABEL: func @_QPnot_test
    ! CHECK: %[[dest:.*]] = fir.alloca i32 {bindc_name = "destination", uniq_name = "_QFnot_testEdestination"}
    ! CHECK: %[[dest_decl:.*]]:2 = hlfir.declare %[[dest]] {uniq_name = "_QFnot_testEdestination"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    ! CHECK: %[[source:.*]] = fir.alloca i32 {bindc_name = "source", uniq_name = "_QFnot_testEsource"}
    ! CHECK: %[[source_decl:.*]]:2 = hlfir.declare %[[source]] {uniq_name = "_QFnot_testEsource"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    ! CHECK: %[[loaded_source:.*]] = fir.load %[[source_decl]]#0 : !fir.ref<i32>
    ! CHECK: %[[all_ones:.*]] = arith.constant -1 : i32
    ! CHECK: %[[result:.*]] = arith.xori %[[loaded_source]], %[[all_ones]] : i32
    ! CHECK: hlfir.assign %[[result]] to %[[dest_decl]]#0 : i32, !fir.ref<i32>
    ! CHECK: return
    destination = not(source)
  end subroutine

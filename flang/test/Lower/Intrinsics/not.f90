! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: not_test
subroutine not_test
    integer :: source
    integer :: destination
    ! CHECK_LABEL: not_test
    ! CHECK: %[[dest:.*]] = fir.alloca i32 {bindc_name = "destination", uniq_name = "_QFnot_testEdestination"}
    ! CHECK: %[[source:.*]] = fir.alloca i32 {bindc_name = "source", uniq_name = "_QFnot_testEsource"}
    ! CHECK: %[[loaded_source:.*]] = fir.load %[[source]] : !fir.ref<i32>
    ! CHECK: %[[all_ones:.*]] = arith.constant -1 : i32
    ! CHECK: %[[result:.*]] = arith.xori %[[loaded_source]], %[[all_ones]] : i32
    ! CHECK: fir.store %[[result]] to %[[dest]] : !fir.ref<i32>
    ! CHECK: return
    destination = not(source)
  end subroutine
  
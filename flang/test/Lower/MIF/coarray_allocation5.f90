! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s

module m_coarray_test
    implicit none
    real :: module_coarray[*]
end module m_coarray_test

program test_module_coarray
    use m_coarray_test
    implicit none
    integer :: me

    me = this_image()
    module_coarray = real(me)

end program test_module_coarray


!CHECK: fir.global @_QMm_coarray_testEmodule_coarray : f32 {
!CHECK:   %[[VAL_0:.*]] = fir.zero_bits f32
!CHECK:   fir.has_value %[[VAL_0]] : f32
!CHECK: }
!CHECK: func.func @_QQmain() attributes {fir.bindc_name = "TEST_MODULE_COARRAY"} {
!CHECK:   %[[VAL_0:.*]] = fir.alloca !fir.array<0xi64>
!CHECK:   %[[VAL_1:.*]] = fir.alloca !fir.array<1xi64>
!CHECK:   %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
!CHECK:   %[[VAL_3:.*]] = fir.alloca i32 {bindc_name = "me", uniq_name = "_QFEme"}
!CHECK:   %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] {uniq_name = "_QFEme"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:   %[[VAL_5:.*]] = fir.address_of(@_QMm_coarray_testEmodule_coarray) : !fir.ref<f32>
!CHECK:   %[[C1_I64:.*]] = arith.constant 1 : i64
!CHECK:   %[[C0:.*]] = arith.constant 0 : index
!CHECK:   %[[VAL_6:.*]] = fir.coordinate_of %[[VAL_1]], %[[C0]] : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
!CHECK:   fir.store %[[C1_I64]] to %[[VAL_6]] : !fir.ref<i64>
!CHECK:   %[[VAL_7:.*]] = fir.embox %[[VAL_1]] : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
!CHECK:   %[[VAL_8:.*]] = fir.embox %[[VAL_0]] : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
!CHECK:   mif.alloc_coarray %[[VAL_5]] lcobounds %[[VAL_7]] ucobounds %[[VAL_8]] {uniq_name = "_QMm_coarray_testEmodule_coarray"} : (!fir.ref<f32>, !fir.box<!fir.array<1xi64>>, !fir.box<!fir.array<0xi64>>) -> ()
!CHECK:   %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_5]] {uniq_name = "_QMm_coarray_testEmodule_coarray"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK:   %[[VAL_10:.*]] = mif.this_image : () -> i32
!CHECK:   hlfir.assign %[[VAL_10]] to %[[VAL_4]]#0 : i32, !fir.ref<i32>
!CHECK:   %[[VAL_11:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i32>
!CHECK:   %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i32) -> f32
!CHECK:   hlfir.assign %[[VAL_12]] to %[[VAL_9]]#0 : f32, !fir.ref<f32>
!CHECK:   return
!CHECK: }


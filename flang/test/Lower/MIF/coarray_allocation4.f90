! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-llvm -fcoarray %s -o - 2>&1 | FileCheck %s --check-prefix=LLVM

! LLVM: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @__mif_save_coarrays_allocate, ptr null }]

subroutine test_coarray_save()
    implicit none
    real, SAVE :: n[*]
    real, SAVE :: m[*]
    real, SAVE :: p[*]
end subroutine test_coarray_save

program main
  call test_coarray_save()
end program

! CHECK-LABEL: func.func @_QPtest_coarray_save()
! CHECK:   %0 = fir.dummy_scope : !fir.dscope
! CHECK:   %1 = fir.address_of(@_QFtest_coarray_saveEm) : !fir.ref<f32>
! CHECK:   %2:2 = hlfir.declare %1 {uniq_name = "_QFtest_coarray_saveEm"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:   %3 = fir.address_of(@_QFtest_coarray_saveEn) : !fir.ref<f32>
! CHECK:   %4:2 = hlfir.declare %3 {uniq_name = "_QFtest_coarray_saveEn"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:   %5 = fir.address_of(@_QFtest_coarray_saveEp) : !fir.ref<f32>
! CHECK:   %6:2 = hlfir.declare %5 {uniq_name = "_QFtest_coarray_saveEp"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:   return

! CHECK-LABEL: func.func @_QQmain() attributes {fir.bindc_name = "MAIN"}
! CHECK:   %0 = fir.dummy_scope : !fir.dscope
! CHECK:   fir.call @_QPtest_coarray_save() fastmath<contract> : () -> ()
! CHECK:   return

! CHECK: fir.global internal @_QFtest_coarray_saveEm : f32

! CHECK-LABEL: func.func @__mif_save_coarrays_allocate()
! CHECK:   %[[VAL_0:.*]] = fir.alloca !fir.array<0xi64>
! CHECK:   %[[VAL_1:.*]] = fir.alloca !fir.array<1xi64>
! CHECK:   %[[VAL_2:.*]] = fir.alloca !fir.array<0xi64>
! CHECK:   %[[VAL_3:.*]] = fir.alloca !fir.array<1xi64>
! CHECK:   %[[VAL_4:.*]] = fir.alloca !fir.array<0xi64>
! CHECK:   %[[VAL_5:.*]] = fir.alloca !fir.array<1xi64>
! CHECK:   %[[INIT_STAT:.*]] = mif.init -> i32
! CHECK:   %[[VAL_6:.*]] = fir.address_of(@_QFtest_coarray_saveEm) : !fir.ref<f32>
! CHECK:   %[[C1_I64:.*]] = arith.constant 1 : i64
! CHECK:   %[[C1_I64_0:.*]] = arith.constant 1 : i64
! CHECK:   %[[C0:.*]] = arith.constant 0 : index
! CHECK:   %[[VAL_7:.*]] = fir.coordinate_of %[[VAL_5]], %[[C0]] : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
! CHECK:   fir.store %[[C1_I64_0]] to %[[VAL_7]] : !fir.ref<i64>
! CHECK:   %[[VAL_8:.*]] = fir.embox %[[VAL_5]] : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
! CHECK:   %[[C1_I64_1:.*]] = arith.constant 1 : i64
! CHECK:   %[[VAL_9:.*]] = fir.embox %[[VAL_4]] : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
! CHECK:   mif.alloc_coarray %[[VAL_6]] lcobounds %[[VAL_8]] ucobounds %[[VAL_9]] {uniq_name = "_QFtest_coarray_saveEm"} : (!fir.ref<f32>, !fir.box<!fir.array<1xi64>>, !fir.box<!fir.array<0xi64>>) -> ()
! CHECK:   %[[VAL_10:.*]] = fir.address_of(@_QFtest_coarray_saveEn) : !fir.ref<f32>
! CHECK:   %[[C1_I64_2:.*]] = arith.constant 1 : i64
! CHECK:   %[[C1_I64_3:.*]] = arith.constant 1 : i64
! CHECK:   %[[C0_4:.*]] = arith.constant 0 : index
! CHECK:   %[[VAL_11:.*]] = fir.coordinate_of %[[VAL_3]], %[[C0_4]] : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
! CHECK:   fir.store %[[C1_I64_3]] to %[[VAL_11]] : !fir.ref<i64>
! CHECK:   %[[VAL_12:.*]] = fir.embox %[[VAL_3]] : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
! CHECK:   %[[C1_I64_5:.*]] = arith.constant 1 : i64
! CHECK:   %[[VAL_13:.*]] = fir.embox %[[VAL_2]] : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
! CHECK:   mif.alloc_coarray %[[VAL_10]] lcobounds %[[VAL_12]] ucobounds %[[VAL_13]] {uniq_name = "_QFtest_coarray_saveEn"} : (!fir.ref<f32>, !fir.box<!fir.array<1xi64>>, !fir.box<!fir.array<0xi64>>) -> ()
! CHECK:   %[[VAL_14:.*]] = fir.address_of(@_QFtest_coarray_saveEp) : !fir.ref<f32>
! CHECK:   %[[C1_I64_6:.*]] = arith.constant 1 : i64
! CHECK:   %[[C1_I64_7:.*]] = arith.constant 1 : i64
! CHECK:   %[[C0_8:.*]] = arith.constant 0 : index
! CHECK:   %[[VAL_15:.*]] = fir.coordinate_of %[[VAL_1]], %[[C0_8]] : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
! CHECK:   fir.store %[[C1_I64_7]] to %[[VAL_15]] : !fir.ref<i64>
! CHECK:   %[[VAL_16:.*]] = fir.embox %[[VAL_1]] : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
! CHECK:   %[[VAL_C1_I64_9:.*]] = arith.constant 1 : i64
! CHECK:   %[[VAL_17:.*]] = fir.embox %[[VAL_0]] : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
! CHECK:   mif.alloc_coarray %[[VAL_14:.*]] lcobounds %[[VAL_16]] ucobounds %[[VAL_17]] {uniq_name = "_QFtest_coarray_saveEp"} : (!fir.ref<f32>, !fir.box<!fir.array<1xi64>>, !fir.box<!fir.array<0xi64>>) -> ()
! CHECK:   return

! CHECK:  fir.global internal @_QFtest_coarray_saveEn : f32
! CHECK:  fir.global internal @_QFtest_coarray_saveEp : f32


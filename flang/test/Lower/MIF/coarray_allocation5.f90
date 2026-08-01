! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-llvm -fcoarray %s -o - 2>&1 | FileCheck %s --check-prefix=LLVM

module m_coarray_test
    implicit none
    real :: module_coarray[*]
end module m_coarray_test

program test
  use m_coarray_test
end program

! LLVM: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @__mif_save_coarrays_allocate, ptr null }]

! CHECK-LABEL: func.func @__mif_save_coarrays_allocate()
! CHECK:  %[[VAL_0:.*]] = fir.alloca !fir.array<0xi64>
! CHECK:  %[[VAL_1:.*]] = fir.alloca !fir.array<1xi64>
! CHECK:  %[[INIT_STAT:.*]] = mif.init -> i32
! CHECK:  %[[VAL_2:.*]] = fir.address_of(@_QMm_coarray_testEmodule_coarray) : !fir.ref<f32>
! CHECK:  %c1_i64 = arith.constant 1 : i64
! CHECK:  %c1_i64_0 = arith.constant 1 : i64
! CHECK:  %c0 = arith.constant 0 : index
! CHECK:  %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_1]], %c0 : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
! CHECK:  fir.store %c1_i64_0 to %[[VAL_3]] : !fir.ref<i64>
! CHECK:  %[[VAL_4:.*]] = fir.embox %[[VAL_1]] : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
! CHECK:  %c1_i64_1 = arith.constant 1 : i64
! CHECK:  %[[VAL_5:.*]] = fir.embox %[[VAL_0]] : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
! CHECK:  mif.alloc_coarray %[[VAL_2]] lcobounds %[[VAL_4]] ucobounds %[[VAL_5]] {uniq_name = "_QMm_coarray_testEmodule_coarray"} : (!fir.ref<f32>, !fir.box<!fir.array<1xi64>>, !fir.box<!fir.array<0xi64>>) -> ()
! CHECK:  return

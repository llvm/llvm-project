! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s

! CHECK: fir.global @_QMtestEcount_i_test : i32
module test
  integer :: count_i_test[*]
end module

! CHECK-LABEL: func.func @_QPtest_save_coarray
subroutine test_save_coarray()
    use test
    implicit none
    integer, save :: count_i[*] = 0
    
    count_i = count_i + 1
    count_i_test = count_i_test + 1

    ! CHECK: %[[VAL_0:.*]] = fir.alloca !fir.array<0xi64>
    ! CHECK: %[[VAL_1:.*]] = fir.alloca !fir.array<1xi64>
    ! CHECK: %[[VAL_2:.*]] = fir.alloca !fir.array<0xi64>
    ! CHECK: %[[VAL_3:.*]] = fir.alloca !fir.array<1xi64>
    ! CHECK: %[[VAL_4:.*]] = fir.dummy_scope : !fir.dscope
    ! CHECK: %[[VAL_5:.*]] = fir.address_of(@_QFtest_save_coarrayEcount_i) : !fir.ref<i32>
    ! CHECK: %[[C1_I64:.*]] = arith.constant 1 : i64
    ! CHECK: %[[C0:.*]] = arith.constant 0 : index
    ! CHECK: %[[VAL_6:.*]] = fir.coordinate_of %[[VAL_3]], %[[C0]] : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
    ! CHECK: fir.store %[[C1_I64]] to %[[VAL_6]] : !fir.ref<i64>
    ! CHECK: %[[VAL_7:.*]] = fir.embox %[[VAL_3]] : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
    ! CHECK: %[[VAL_8:.*]] = fir.embox %[[VAL_2]] : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
    ! CHECK: mif.alloc_coarray %[[VAL_5]] lcobounds %[[VAL_7]] ucobounds %[[VAL_8]] {uniq_name = "_QFtest_save_coarrayEcount_i"} : (!fir.ref<i32>, !fir.box<!fir.array<1xi64>>, !fir.box<!fir.array<0xi64>>) -> ()
    ! CHECK: %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_5]] {uniq_name = "_QFtest_save_coarrayEcount_i"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    ! CHECK: %[[VAL_10:.*]] = fir.address_of(@_QMtestEcount_i_test) : !fir.ref<i32>
    ! CHECK: %[[C1_I64_0:.*]] = arith.constant 1 : i64
    ! CHECK: %[[C0_1:.*]] = arith.constant 0 : index
    ! CHECK: %[[VAL_11:.*]] = fir.coordinate_of %[[VAL_1]], %[[C0_1]] : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
    ! CHECK: fir.store %[[C1_I64_0]] to %[[VAL_11]] : !fir.ref<i64>
    ! CHECK: %[[VAL_12:.*]] = fir.embox %[[VAL_1]] : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
    ! CHECK: %[[VAL_13:.*]] = fir.embox %0 : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
    ! CHECK: mif.alloc_coarray %[[VAL_10]] lcobounds %[[VAL_12]] ucobounds %[[VAL_13]] {uniq_name = "_QMtestEcount_i_test"} : (!fir.ref<i32>, !fir.box<!fir.array<1xi64>>, !fir.box<!fir.array<0xi64>>) -> ()
    ! CHECK: %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_10]] {uniq_name = "_QMtestEcount_i_test"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    ! CHECK: %[[VAL_15:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<i32>
    ! CHECK: %[[C1_I32:.*]] = arith.constant 1 : i32
    ! CHECK: %[[VAL_16:.*]] = arith.addi %[[VAL_15]], %[[C1_I32]] : i32
    ! CHECK: hlfir.assign %[[VAL_16]] to %[[VAL_9]]#0 : i32, !fir.ref<i32>
    ! CHECK: %[[VAL_17:.*]] = fir.load %[[VAL_14]]#0 : !fir.ref<i32>
    ! CHECK: %[[C1_I32_2:.*]] = arith.constant 1 : i32
    ! CHECK: %[[VAL_18:.*]] = arith.addi %[[VAL_17]], %[[C1_I32_2]] : i32
    ! CHECK: hlfir.assign %[[VAL_18]] to %[[VAL_14]]#0 : i32, !fir.ref<i32>
    ! CHECK: mif.dealloc_coarray %[[VAL_14]]#0 : (!fir.ref<i32>) -> ()
    ! CHECK: mif.dealloc_coarray %[[VAL_9]]#0 : (!fir.ref<i32>) -> ()
end subroutine test_save_coarray

! CHECK-LABEL: func.func @_QQmain
program main
  use test
  
  call test_save_coarray()
  ! CHECK: %[[VAL_0:.*]] = fir.alloca !fir.array<0xi64>
  ! CHECK: %[[VAL_1:.*]] = fir.alloca !fir.array<1xi64>
  ! CHECK: %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[VAL_3:.*]] = fir.address_of(@_QMtestEcount_i_test) : !fir.ref<i32>
  ! CHECK: %[[C1_I64:.*]] = arith.constant 1 : i64
  ! CHECK: %[[C0:.*]] = arith.constant 0 : index
  ! CHECK: %[[VAL_4:.*]] = fir.coordinate_of %[[VAL_1]], %[[C0]] : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
  ! CHECK: fir.store %[[C1_I64]] to %[[VAL_4]] : !fir.ref<i64>
  ! CHECK: %[[VAL_5:.*]] = fir.embox %[[VAL_1]] : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
  ! CHECK: %[[VAL_6:.*]] = fir.embox %[[VAL_0]] : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
  ! CHECK: mif.alloc_coarray %[[VAL_3]] lcobounds %[[VAL_5]] ucobounds %[[VAL_6]] {uniq_name = "_QMtestEcount_i_test"} : (!fir.ref<i32>, !fir.box<!fir.array<1xi64>>, !fir.box<!fir.array<0xi64>>) -> ()
  ! CHECK: %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_3]] {uniq_name = "_QMtestEcount_i_test"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK: fir.call @_QPtest_save_coarray() fastmath<contract> : () -> ()
  ! CHECK: mif.dealloc_coarray %[[VAL_7]]#0 : (!fir.ref<i32>) -> ()
  ! CHECK: return

end program main


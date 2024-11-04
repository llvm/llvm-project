! Test shape lowering for assumed-rank
! RUN: bbc -emit-hlfir -o - %s -allow-assumed-rank | FileCheck %s

subroutine test_shape(x)
  real :: x(..)
  call takes_integer_array(shape(x))
end subroutine
! CHECK-LABEL:   func.func @_QPtest_shape(
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.array<15xi32>
! CHECK:           %[[VAL_4:.*]] = arith.constant 4 : i32
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.array<15xi32>>) -> !fir.llvm_ptr<i8>
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_3:.*]] : (!fir.box<!fir.array<*:f32>>) -> !fir.box<none>
! CHECK:           %[[VAL_10:.*]] = fir.call @_FortranAShape(%[[VAL_7]], %[[VAL_8]], %[[VAL_4]], %{{.*}}, %{{.*}})
! CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.array<15xi32>>) -> !fir.ref<!fir.array<?xi32>>
! CHECK:           %[[VAL_12:.*]] = fir.box_rank %[[VAL_3]] : (!fir.box<!fir.array<*:f32>>) -> index
! CHECK:           %[[VAL_13:.*]] = fir.shape %[[VAL_12]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_11]](%[[VAL_13]]) {uniq_name = ".tmp.intrinsic_result"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>)
! CHECK:           %[[VAL_15:.*]] = arith.constant false
! CHECK:           %[[VAL_16:.*]] = hlfir.as_expr %[[VAL_14]]#0 move %[[VAL_15]] : (!fir.box<!fir.array<?xi32>>, i1) -> !hlfir.expr<?xi32>
! CHECK:           %[[VAL_17:.*]]:3 = hlfir.associate %[[VAL_16]](%[[VAL_13]]) {adapt.valuebyref} : (!hlfir.expr<?xi32>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>, i1)
! CHECK:           fir.call @_QPtakes_integer_array(%[[VAL_17]]#1) fastmath<contract> : (!fir.ref<!fir.array<?xi32>>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_17]]#1, %[[VAL_17]]#2 : !fir.ref<!fir.array<?xi32>>, i1
! CHECK:           hlfir.destroy %[[VAL_16]] : !hlfir.expr<?xi32>
! CHECK:           return
! CHECK:         }

subroutine test_shape_kind(x)
  real :: x(..)
  call takes_integer8_array(shape(x, kind=8))
end subroutine
! CHECK-LABEL:   func.func @_QPtest_shape_kind(
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.array<15xi64>
! CHECK:           %[[VAL_4:.*]] = arith.constant 8 : i32
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.array<15xi64>>) -> !fir.llvm_ptr<i8>
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_3:.*]] : (!fir.box<!fir.array<*:f32>>) -> !fir.box<none>
! CHECK:           %[[VAL_10:.*]] = fir.call @_FortranAShape(%[[VAL_7]], %[[VAL_8]], %[[VAL_4]], %{{.*}}, %{{.*}})
! CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.array<15xi64>>) -> !fir.ref<!fir.array<?xi64>>
! CHECK:           %[[VAL_12:.*]] = fir.box_rank %[[VAL_3]] : (!fir.box<!fir.array<*:f32>>) -> index
! CHECK:           %[[VAL_13:.*]] = fir.shape %[[VAL_12]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_11]](%[[VAL_13]]) {uniq_name = ".tmp.intrinsic_result"} : (!fir.ref<!fir.array<?xi64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi64>>, !fir.ref<!fir.array<?xi64>>)

subroutine test_shape_2(x)
  real, pointer :: x(..)
  call takes_integer_array(shape(x))
end subroutine
! CHECK-LABEL:   func.func @_QPtest_shape_2(
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.array<15xi32>
! CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_3:.*]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>
! CHECK:           %[[VAL_5:.*]] = arith.constant 4 : i32
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.array<15xi32>>) -> !fir.llvm_ptr<i8>
! CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_4]] : (!fir.box<!fir.ptr<!fir.array<*:f32>>>) -> !fir.box<none>
! CHECK:           %[[VAL_11:.*]] = fir.call @_FortranAShape(%[[VAL_8]], %[[VAL_9]], %[[VAL_5]], %{{.*}}, %{{.*}})
! CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.array<15xi32>>) -> !fir.ref<!fir.array<?xi32>>
! CHECK:           %[[VAL_13:.*]] = fir.box_rank %[[VAL_4]] : (!fir.box<!fir.ptr<!fir.array<*:f32>>>) -> index
! CHECK:           %[[VAL_14:.*]] = fir.shape %[[VAL_13]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_12]](%[[VAL_14]]) {uniq_name = ".tmp.intrinsic_result"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>)

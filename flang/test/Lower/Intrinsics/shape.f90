! Test SHAPE with function results
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

subroutine test()
  interface
    function return_array()
      real, pointer :: return_array(:, :, :)
    end function
  end interface
  print *, shape(return_array())
end subroutine
! CHECK-LABEL:   func.func @_QPtest() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.array<3xi32>
! CHECK:           %[[VAL_7:.*]] = fir.call @_QPreturn_array() {{.*}}: () -> !fir.box<!fir.ptr<!fir.array<?x?x?xf32>>>
! CHECK:           fir.save_result %[[VAL_7]] to %[[VAL_1:.*]] : !fir.box<!fir.ptr<!fir.array<?x?x?xf32>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x?xf32>>>>
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x?xf32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x?xf32>>>>)
! CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_8]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x?xf32>>>>
! CHECK:           %[[VAL_10:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_11:.*]]:3 = fir.box_dims %[[VAL_9]], %[[VAL_10]] : (!fir.box<!fir.ptr<!fir.array<?x?x?xf32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_11]]#1 : (index) -> i32
! CHECK:           %[[VAL_13:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_14:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_13]] : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
! CHECK:           fir.store %[[VAL_12]] to %[[VAL_14]] : !fir.ref<i32>
! CHECK:           %[[VAL_15:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_16:.*]]:3 = fir.box_dims %[[VAL_9]], %[[VAL_15]] : (!fir.box<!fir.ptr<!fir.array<?x?x?xf32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_16]]#1 : (index) -> i32
! CHECK:           %[[VAL_18:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_19:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_18]] : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
! CHECK:           fir.store %[[VAL_17]] to %[[VAL_19]] : !fir.ref<i32>
! CHECK:           %[[VAL_20:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_21:.*]]:3 = fir.box_dims %[[VAL_9]], %[[VAL_20]] : (!fir.box<!fir.ptr<!fir.array<?x?x?xf32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_21]]#1 : (index) -> i32
! CHECK:           %[[VAL_23:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_24:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_23]] : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
! CHECK:           fir.store %[[VAL_22]] to %[[VAL_24]] : !fir.ref<i32>
! CHECK:           %[[VAL_25:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_26:.*]] = fir.shape %[[VAL_25]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_27:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_26]]) {uniq_name = ".tmp.intrinsic_result"} : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi32>>, !fir.ref<!fir.array<3xi32>>)

subroutine test_kind()
  interface
    function return_array()
      real, pointer :: return_array(:, :, :)
    end function
  end interface
  print *, shape(return_array(), kind=8)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_kind() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.array<3xi64>
! CHECK:           %[[VAL_7:.*]] = fir.call @_QPreturn_array() {{.*}}: () -> !fir.box<!fir.ptr<!fir.array<?x?x?xf32>>>
! CHECK:           fir.save_result %[[VAL_7]] to %[[VAL_1:.*]] : !fir.box<!fir.ptr<!fir.array<?x?x?xf32>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x?xf32>>>>
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x?xf32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x?xf32>>>>)
! CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_8]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x?xf32>>>>
! CHECK:           %[[VAL_10:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_11:.*]]:3 = fir.box_dims %[[VAL_9]], %[[VAL_10]] : (!fir.box<!fir.ptr<!fir.array<?x?x?xf32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_11]]#1 : (index) -> i64
! CHECK:           %[[VAL_13:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_14:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_13]] : (!fir.ref<!fir.array<3xi64>>, index) -> !fir.ref<i64>
! CHECK:           fir.store %[[VAL_12]] to %[[VAL_14]] : !fir.ref<i64>
! CHECK:           %[[VAL_15:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_16:.*]]:3 = fir.box_dims %[[VAL_9]], %[[VAL_15]] : (!fir.box<!fir.ptr<!fir.array<?x?x?xf32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_16]]#1 : (index) -> i64
! CHECK:           %[[VAL_18:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_19:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_18]] : (!fir.ref<!fir.array<3xi64>>, index) -> !fir.ref<i64>
! CHECK:           fir.store %[[VAL_17]] to %[[VAL_19]] : !fir.ref<i64>
! CHECK:           %[[VAL_20:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_21:.*]]:3 = fir.box_dims %[[VAL_9]], %[[VAL_20]] : (!fir.box<!fir.ptr<!fir.array<?x?x?xf32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_21]]#1 : (index) -> i64
! CHECK:           %[[VAL_23:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_24:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_23]] : (!fir.ref<!fir.array<3xi64>>, index) -> !fir.ref<i64>
! CHECK:           fir.store %[[VAL_22]] to %[[VAL_24]] : !fir.ref<i64>
! CHECK:           %[[VAL_25:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_26:.*]] = fir.shape %[[VAL_25]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_27:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_26]]) {uniq_name = ".tmp.intrinsic_result"} : (!fir.ref<!fir.array<3xi64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi64>>, !fir.ref<!fir.array<3xi64>>)

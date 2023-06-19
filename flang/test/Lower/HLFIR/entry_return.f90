! RUN: bbc -emit-hlfir -o - %s | FileCheck %s
! test multiple return values with different types coming from ENTRY statements

complex function f1()
  logical e1
  entry e1()
  e1 = .false.
end function
! CHECK-LABEL:   func.func @_QPf1() -> !fir.complex<4> {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.complex<4> {bindc_name = "f1", uniq_name = "_QFf1Ef1"}
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFf1Ef1"} : (!fir.ref<!fir.complex<4>>) -> (!fir.ref<!fir.complex<4>>, !fir.ref<!fir.complex<4>>)
! CHECK:           %[[VAL_2:.*]] = fir.convert %[[VAL_1]]#1 : (!fir.ref<!fir.complex<4>>) -> !fir.ref<!fir.logical<4>>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFf1Ee1"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:           cf.br ^bb1
! CHECK:         ^bb1:
! CHECK:           %[[VAL_4:.*]] = arith.constant false
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i1) -> !fir.logical<4>
! CHECK:           hlfir.assign %[[VAL_5]] to %[[VAL_3]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
! CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_1]]#1 : !fir.ref<!fir.complex<4>>
! CHECK:           return %[[VAL_6]] : !fir.complex<4>
! CHECK:         }

! // CHECK-LABEL:   func.func @_QPe1() -> !fir.logical<4> {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.complex<4> {bindc_name = "f1", uniq_name = "_QFf1Ef1"}
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFf1Ef1"} : (!fir.ref<!fir.complex<4>>) -> (!fir.ref<!fir.complex<4>>, !fir.ref<!fir.complex<4>>)
! CHECK:           %[[VAL_2:.*]] = fir.convert %[[VAL_1]]#1 : (!fir.ref<!fir.complex<4>>) -> !fir.ref<!fir.logical<4>>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFf1Ee1"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:           cf.br ^bb1
! CHECK:         ^bb1:
! CHECK:           %[[VAL_4:.*]] = arith.constant false
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i1) -> !fir.logical<4>
! CHECK:           hlfir.assign %[[VAL_5]] to %[[VAL_3]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
! CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_3]]#1 : !fir.ref<!fir.logical<4>>
! CHECK:           return %[[VAL_6]] : !fir.logical<4>
! CHECK:         }

logical function f2()
  complex e2
  entry e2()
  e2 = complex(1.0, 2.0)
end function
! CHECK-LABEL:   func.func @_QPf2() -> !fir.logical<4> {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.complex<4> {bindc_name = "e2", uniq_name = "_QFf2Ee2"}
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFf2Ee2"} : (!fir.ref<!fir.complex<4>>) -> (!fir.ref<!fir.complex<4>>, !fir.ref<!fir.complex<4>>)
! CHECK:           %[[VAL_2:.*]] = fir.convert %[[VAL_1]]#1 : (!fir.ref<!fir.complex<4>>) -> !fir.ref<!fir.logical<4>>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFf2Ef2"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:           cf.br ^bb1
! CHECK:         ^bb1:
! CHECK:           %[[VAL_4:.*]] = arith.constant 1.000000e+00 : f32
! CHECK:           %[[VAL_5:.*]] = arith.constant 2.000000e+00 : f32
! CHECK:           %[[VAL_6:.*]]:3 = hlfir.associate %[[VAL_4]] {uniq_name = "adapt.valuebyref"} : (f32) -> (!fir.ref<f32>, !fir.ref<f32>, i1)
! CHECK:           %[[VAL_7:.*]]:3 = hlfir.associate %[[VAL_5]] {uniq_name = "adapt.valuebyref"} : (f32) -> (!fir.ref<f32>, !fir.ref<f32>, i1)
! CHECK:           %[[VAL_8:.*]] = fir.call @_QPcomplex(%[[VAL_6]]#1, %[[VAL_7]]#1) fastmath<contract> : (!fir.ref<f32>, !fir.ref<f32>) -> f32
! CHECK:           hlfir.end_associate %[[VAL_6]]#1, %[[VAL_6]]#2 : !fir.ref<f32>, i1
! CHECK:           hlfir.end_associate %[[VAL_7]]#1, %[[VAL_7]]#2 : !fir.ref<f32>, i1
! CHECK:           %[[VAL_9:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:           %[[VAL_10:.*]] = fir.undefined !fir.complex<4>
! CHECK:           %[[VAL_11:.*]] = fir.insert_value %[[VAL_10]], %[[VAL_8]], [0 : index] : (!fir.complex<4>, f32) -> !fir.complex<4>
! CHECK:           %[[VAL_12:.*]] = fir.insert_value %[[VAL_11]], %[[VAL_9]], [1 : index] : (!fir.complex<4>, f32) -> !fir.complex<4>
! CHECK:           hlfir.assign %[[VAL_12]] to %[[VAL_1]]#0 : !fir.complex<4>, !fir.ref<!fir.complex<4>>
! CHECK:           %[[VAL_13:.*]] = fir.load %[[VAL_3]]#1 : !fir.ref<!fir.logical<4>>
! CHECK:           return %[[VAL_13]] : !fir.logical<4>
! CHECK:         }

! CHECK-LABEL:   func.func @_QPe2() -> !fir.complex<4> {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.complex<4> {bindc_name = "e2", uniq_name = "_QFf2Ee2"}
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFf2Ee2"} : (!fir.ref<!fir.complex<4>>) -> (!fir.ref<!fir.complex<4>>, !fir.ref<!fir.complex<4>>)
! CHECK:           %[[VAL_2:.*]] = fir.convert %[[VAL_1]]#1 : (!fir.ref<!fir.complex<4>>) -> !fir.ref<!fir.logical<4>>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFf2Ef2"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:           cf.br ^bb1
! CHECK:         ^bb1:
! CHECK:           %[[VAL_4:.*]] = arith.constant 1.000000e+00 : f32
! CHECK:           %[[VAL_5:.*]] = arith.constant 2.000000e+00 : f32
! CHECK:           %[[VAL_6:.*]]:3 = hlfir.associate %[[VAL_4]] {uniq_name = "adapt.valuebyref"} : (f32) -> (!fir.ref<f32>, !fir.ref<f32>, i1)
! CHECK:           %[[VAL_7:.*]]:3 = hlfir.associate %[[VAL_5]] {uniq_name = "adapt.valuebyref"} : (f32) -> (!fir.ref<f32>, !fir.ref<f32>, i1)
! CHECK:           %[[VAL_8:.*]] = fir.call @_QPcomplex(%[[VAL_6]]#1, %[[VAL_7]]#1) fastmath<contract> : (!fir.ref<f32>, !fir.ref<f32>) -> f32
! CHECK:           hlfir.end_associate %[[VAL_6]]#1, %[[VAL_6]]#2 : !fir.ref<f32>, i1
! CHECK:           hlfir.end_associate %[[VAL_7]]#1, %[[VAL_7]]#2 : !fir.ref<f32>, i1
! CHECK:           %[[VAL_9:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:           %[[VAL_10:.*]] = fir.undefined !fir.complex<4>
! CHECK:           %[[VAL_11:.*]] = fir.insert_value %[[VAL_10]], %[[VAL_8]], [0 : index] : (!fir.complex<4>, f32) -> !fir.complex<4>
! CHECK:           %[[VAL_12:.*]] = fir.insert_value %[[VAL_11]], %[[VAL_9]], [1 : index] : (!fir.complex<4>, f32) -> !fir.complex<4>
! CHECK:           hlfir.assign %[[VAL_12]] to %[[VAL_1]]#0 : !fir.complex<4>, !fir.ref<!fir.complex<4>>
! CHECK:           %[[VAL_13:.*]] = fir.load %[[VAL_1]]#1 : !fir.ref<!fir.complex<4>>
! CHECK:           return %[[VAL_13]] : !fir.complex<4>
! CHECK:         }

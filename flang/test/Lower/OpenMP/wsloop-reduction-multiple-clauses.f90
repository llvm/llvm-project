! RUN: bbc -emit-hlfir -fopenmp -o - %s | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s | FileCheck %s

program main
  implicit none

  integer:: i
  REAL(8) :: scalar
  REAL(8) :: array(3,3)

  scalar = 0
  array = 0

  !$omp parallel do reduction(+:scalar) reduction(+:array)
  do i=1,10
    scalar = i
    array(1, 1) = i + 1
    array(2, 2) = i + 2
    array(3, 3) = i + 3
  enddo

  print *,scalar
  print *,array

endprogram

! CHECK-LABEL:   omp.declare_reduction @add_reduction_byref_box_3x3xf64 : !fir.ref<!fir.box<!fir.array<3x3xf64>>> alloc {
! CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.array<3x3xf64>>
! CHECK:           omp.yield(%[[VAL_3]] : !fir.ref<!fir.box<!fir.array<3x3xf64>>>)
! CHECK-LABEL:   } init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.array<3x3xf64>>>, %[[ALLOC:.*]]: !fir.ref<!fir.box<!fir.array<3x3xf64>>>):
! CHECK:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f64
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.array<3x3xf64>>>
! CHECK:           %[[VAL_4:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_5:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_4]], %[[VAL_5]] : (index, index) -> !fir.shape<2>
! CHECK:           %[[VAL_7:.*]] = fir.allocmem !fir.array<3x3xf64> {bindc_name = ".tmp", uniq_name = ""}
! CHECK:           %[[VAL_8:.*]] = arith.constant true
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_7]](%[[VAL_6]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<3x3xf64>>, !fir.shape<2>) -> (!fir.heap<!fir.array<3x3xf64>>, !fir.heap<!fir.array<3x3xf64>>)
! CHECK:           %[[VAL_10:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_11:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_10]] : (!fir.box<!fir.array<3x3xf64>>, index) -> (index, index, index)
! CHECK:           %[[VAL_12:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_13:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_12]] : (!fir.box<!fir.array<3x3xf64>>, index) -> (index, index, index)
! CHECK:           %[[VAL_14:.*]] = fir.shape_shift %[[VAL_11]]#0, %[[VAL_11]]#1, %[[VAL_13]]#0, %[[VAL_13]]#1 : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK:           %[[VAL_15:.*]] = fir.embox %[[VAL_9]]#0(%[[VAL_14]]) : (!fir.heap<!fir.array<3x3xf64>>, !fir.shapeshift<2>) -> !fir.box<!fir.array<3x3xf64>>
! CHECK:           hlfir.assign %[[VAL_1]] to %[[VAL_15]] : f64, !fir.box<!fir.array<3x3xf64>>
! CHECK:           fir.store %[[VAL_15]] to %[[ALLOC]] : !fir.ref<!fir.box<!fir.array<3x3xf64>>>
! CHECK:           omp.yield(%[[ALLOC]] : !fir.ref<!fir.box<!fir.array<3x3xf64>>>)

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.array<3x3xf64>>>, %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.array<3x3xf64>>>):
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.array<3x3xf64>>>
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.array<3x3xf64>>>
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_4]] : (!fir.box<!fir.array<3x3xf64>>, index) -> (index, index, index)
! CHECK:           %[[VAL_6:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_7:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_6]] : (!fir.box<!fir.array<3x3xf64>>, index) -> (index, index, index)
! CHECK:           %[[VAL_8:.*]] = fir.shape_shift %[[VAL_5]]#0, %[[VAL_5]]#1, %[[VAL_7]]#0, %[[VAL_7]]#1 : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK:           %[[VAL_9:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_10:.*]] = %[[VAL_9]] to %[[VAL_7]]#1 step %[[VAL_9]] unordered {
! CHECK:             fir.do_loop %[[VAL_11:.*]] = %[[VAL_9]] to %[[VAL_5]]#1 step %[[VAL_9]] unordered {
! CHECK:               %[[VAL_12:.*]] = fir.array_coor %[[VAL_2]](%[[VAL_8]]) %[[VAL_11]], %[[VAL_10]] : (!fir.box<!fir.array<3x3xf64>>, !fir.shapeshift<2>, index, index) -> !fir.ref<f64>
! CHECK:               %[[VAL_13:.*]] = fir.array_coor %[[VAL_3]](%[[VAL_8]]) %[[VAL_11]], %[[VAL_10]] : (!fir.box<!fir.array<3x3xf64>>, !fir.shapeshift<2>, index, index) -> !fir.ref<f64>
! CHECK:               %[[VAL_14:.*]] = fir.load %[[VAL_12]] : !fir.ref<f64>
! CHECK:               %[[VAL_15:.*]] = fir.load %[[VAL_13]] : !fir.ref<f64>
! CHECK:               %[[VAL_16:.*]] = arith.addf %[[VAL_14]], %[[VAL_15]] fastmath<contract> : f64
! CHECK:               fir.store %[[VAL_16]] to %[[VAL_12]] : !fir.ref<f64>
! CHECK:             }
! CHECK:           }
! CHECK:           omp.yield(%[[VAL_0]] : !fir.ref<!fir.box<!fir.array<3x3xf64>>>)

! CHECK-LABEL:   }  cleanup {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.array<3x3xf64>>>):
! CHECK:           %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.array<3x3xf64>>>
! CHECK:           %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.array<3x3xf64>>) -> !fir.ref<!fir.array<3x3xf64>>
! CHECK:           %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.array<3x3xf64>>) -> i64
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_5:.*]] = arith.cmpi ne, %[[VAL_3]], %[[VAL_4]] : i64
! CHECK:           fir.if %[[VAL_5]] {
! CHECK:             %[[VAL_6:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.array<3x3xf64>>) -> !fir.heap<!fir.array<3x3xf64>>
! CHECK:             fir.freemem %[[VAL_6]] : !fir.heap<!fir.array<3x3xf64>>
! CHECK:           }
! CHECK:           omp.yield
! CHECK:         }

! CHECK-LABEL:   omp.declare_reduction @add_reduction_f64 : f64 init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: f64):
! CHECK:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f64
! CHECK:           omp.yield(%[[VAL_1]] : f64)

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: f64, %[[VAL_1:.*]]: f64):
! CHECK:           %[[VAL_2:.*]] = arith.addf %[[VAL_0]], %[[VAL_1]] fastmath<contract> : f64
! CHECK:           omp.yield(%[[VAL_2]] : f64)
! CHECK:         }

! CHECK-LABEL:   func.func @_QQmain() attributes {fir.bindc_name = "main"} {
! CHECK:           %[[VAL_0:.*]] = fir.address_of(@_QFEarray) : !fir.ref<!fir.array<3x3xf64>>
! CHECK:           %[[VAL_1:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_2:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_1]], %[[VAL_2]] : (index, index) -> !fir.shape<2>
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_3]]) {uniq_name = "_QFEarray"} : (!fir.ref<!fir.array<3x3xf64>>, !fir.shape<2>) -> (!fir.ref<!fir.array<3x3xf64>>, !fir.ref<!fir.array<3x3xf64>>)
! CHECK:           %[[VAL_5:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_7:.*]] = fir.alloca f64 {bindc_name = "scalar", uniq_name = "_QFEscalar"}
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]] {uniq_name = "_QFEscalar"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
! CHECK:           %[[VAL_9:.*]] = arith.constant 0.000000e+00 : f64
! CHECK:           hlfir.assign %[[VAL_9]] to %[[VAL_8]]#0 : f64, !fir.ref<f64>
! CHECK:           %[[VAL_10:.*]] = arith.constant 0.000000e+00 : f64
! CHECK:           hlfir.assign %[[VAL_10]] to %[[VAL_4]]#0 : f64, !fir.ref<!fir.array<3x3xf64>>
! CHECK:           omp.parallel {
! CHECK:             %[[VAL_11:.*]] = fir.embox %[[VAL_4]]#0(%[[VAL_3]]) : (!fir.ref<!fir.array<3x3xf64>>, !fir.shape<2>) -> !fir.box<!fir.array<3x3xf64>>
! CHECK:             %[[VAL_12:.*]] = fir.alloca !fir.box<!fir.array<3x3xf64>>
! CHECK:             fir.store %[[VAL_11]] to %[[VAL_12]] : !fir.ref<!fir.box<!fir.array<3x3xf64>>>
! CHECK:             %[[VAL_15:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_16:.*]] = arith.constant 10 : i32
! CHECK:             %[[VAL_17:.*]] = arith.constant 1 : i32
! CHECK:             omp.wsloop private(@{{.*}} %{{.*}}#0 -> %[[VAL_13:.*]] : !fir.ref<i32>) reduction(@add_reduction_f64 %[[VAL_8]]#0 -> %[[VAL_18:.*]], byref @add_reduction_byref_box_3x3xf64 %[[VAL_12]] -> %[[VAL_19:.*]] : !fir.ref<f64>, !fir.ref<!fir.box<!fir.array<3x3xf64>>>) {
! CHECK:               omp.loop_nest (%[[VAL_20:.*]]) : i32 = (%[[VAL_15]]) to (%[[VAL_16]]) inclusive step (%[[VAL_17]]) {
! CHECK:                 %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_13]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:                 %[[VAL_21:.*]]:2 = hlfir.declare %[[VAL_18]] {uniq_name = "_QFEscalar"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
! CHECK:                 %[[VAL_22:.*]]:2 = hlfir.declare %[[VAL_19]] {uniq_name = "_QFEarray"} : (!fir.ref<!fir.box<!fir.array<3x3xf64>>>) -> (!fir.ref<!fir.box<!fir.array<3x3xf64>>>, !fir.ref<!fir.box<!fir.array<3x3xf64>>>)
! CHECK:                 fir.store %[[VAL_20]] to %[[VAL_14]]#1 : !fir.ref<i32>
! CHECK:                 %[[VAL_23:.*]] = fir.load %[[VAL_14]]#0 : !fir.ref<i32>
! CHECK:                 %[[VAL_24:.*]] = fir.convert %[[VAL_23]] : (i32) -> f64
! CHECK:                 hlfir.assign %[[VAL_24]] to %[[VAL_21]]#0 : f64, !fir.ref<f64>
! CHECK:                 %[[VAL_25:.*]] = fir.load %[[VAL_14]]#0 : !fir.ref<i32>
! CHECK:                 %[[VAL_26:.*]] = arith.constant 1 : i32
! CHECK:                 %[[VAL_27:.*]] = arith.addi %[[VAL_25]], %[[VAL_26]] : i32
! CHECK:                 %[[VAL_28:.*]] = fir.convert %[[VAL_27]] : (i32) -> f64
! CHECK:                 %[[VAL_29:.*]] = fir.load %[[VAL_22]]#0 : !fir.ref<!fir.box<!fir.array<3x3xf64>>>
! CHECK:                 %[[VAL_30:.*]] = arith.constant 1 : index
! CHECK:                 %[[VAL_31:.*]] = arith.constant 1 : index
! CHECK:                 %[[VAL_32:.*]] = hlfir.designate %[[VAL_29]] (%[[VAL_30]], %[[VAL_31]])  : (!fir.box<!fir.array<3x3xf64>>, index, index) -> !fir.ref<f64>
! CHECK:                 hlfir.assign %[[VAL_28]] to %[[VAL_32]] : f64, !fir.ref<f64>
! CHECK:                 %[[VAL_33:.*]] = fir.load %[[VAL_14]]#0 : !fir.ref<i32>
! CHECK:                 %[[VAL_34:.*]] = arith.constant 2 : i32
! CHECK:                 %[[VAL_35:.*]] = arith.addi %[[VAL_33]], %[[VAL_34]] : i32
! CHECK:                 %[[VAL_36:.*]] = fir.convert %[[VAL_35]] : (i32) -> f64
! CHECK:                 %[[VAL_37:.*]] = fir.load %[[VAL_22]]#0 : !fir.ref<!fir.box<!fir.array<3x3xf64>>>
! CHECK:                 %[[VAL_38:.*]] = arith.constant 2 : index
! CHECK:                 %[[VAL_39:.*]] = arith.constant 2 : index
! CHECK:                 %[[VAL_40:.*]] = hlfir.designate %[[VAL_37]] (%[[VAL_38]], %[[VAL_39]])  : (!fir.box<!fir.array<3x3xf64>>, index, index) -> !fir.ref<f64>
! CHECK:                 hlfir.assign %[[VAL_36]] to %[[VAL_40]] : f64, !fir.ref<f64>
! CHECK:                 %[[VAL_41:.*]] = fir.load %[[VAL_14]]#0 : !fir.ref<i32>
! CHECK:                 %[[VAL_42:.*]] = arith.constant 3 : i32
! CHECK:                 %[[VAL_43:.*]] = arith.addi %[[VAL_41]], %[[VAL_42]] : i32
! CHECK:                 %[[VAL_44:.*]] = fir.convert %[[VAL_43]] : (i32) -> f64
! CHECK:                 %[[VAL_45:.*]] = fir.load %[[VAL_22]]#0 : !fir.ref<!fir.box<!fir.array<3x3xf64>>>
! CHECK:                 %[[VAL_46:.*]] = arith.constant 3 : index
! CHECK:                 %[[VAL_47:.*]] = arith.constant 3 : index
! CHECK:                 %[[VAL_48:.*]] = hlfir.designate %[[VAL_45]] (%[[VAL_46]], %[[VAL_47]])  : (!fir.box<!fir.array<3x3xf64>>, index, index) -> !fir.ref<f64>
! CHECK:                 hlfir.assign %[[VAL_44]] to %[[VAL_48]] : f64, !fir.ref<f64>
! CHECK:                 omp.yield
! CHECK:               }
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }

! CHECK-LABEL:   fir.global internal @_QFEarray : !fir.array<3x3xf64> {
! CHECK:           %[[VAL_0:.*]] = fir.zero_bits !fir.array<3x3xf64>
! CHECK:           fir.has_value %[[VAL_0]] : !fir.array<3x3xf64>
! CHECK:         }
! CHECK:         func.func private @_FortranAioBeginExternalListOutput(i32, !fir.ref<i8>, i32) -> !fir.ref<i8> attributes {fir.io, fir.runtime}

! RUN: bbc -emit-hlfir -o - %s -I nowhere | FileCheck %s

subroutine test1
  interface
     function array_func1(x)
       real:: x, array_func1(10)
     end function array_func1
  end interface
  real :: x(10)
  x = array_func1(1.0)
end subroutine test1
! CHECK-LABEL:   func.func @_QPtest1() {
! CHECK:           %[[VAL_5:.*]] = arith.constant 1.000000e+00 : f32
! CHECK:           %[[VAL_6:.*]]:3 = hlfir.associate %[[VAL_5]] {adapt.valuebyref} : (f32) -> (!fir.ref<f32>, !fir.ref<f32>, i1)
! CHECK:           %[[VAL_17:.*]] = hlfir.eval_in_mem shape %{{.*}} : (!fir.shape<1>) -> !hlfir.expr<10xf32> {
! CHECK:             fir.call @_QParray_func1
! CHECK:             fir.save_result
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_17]] to %{{.*}} : !hlfir.expr<10xf32>, !fir.ref<!fir.array<10xf32>>
! CHECK:           hlfir.end_associate %[[VAL_6]]#1, %[[VAL_6]]#2 : !fir.ref<f32>, i1

subroutine test2(x)
  interface
     function array_func2(x,y)
       real:: x(*), array_func2(10), y
     end function array_func2
  end interface
  real :: x(:)
  x = array_func2(x, 1.0)
end subroutine test2
! CHECK-LABEL:   func.func @_QPtest2(
! CHECK:           %[[VAL_3:.*]] = arith.constant 1.000000e+00 : f32
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.copy_in %{{.*}} to %{{.*}} : (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.box<!fir.array<?xf32>>, i1)
! CHECK:           %[[VAL_5:.*]] = fir.box_addr %[[VAL_4]]#0 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:           %[[VAL_6:.*]]:3 = hlfir.associate %[[VAL_3]] {adapt.valuebyref} : (f32) -> (!fir.ref<f32>, !fir.ref<f32>, i1)
! CHECK:           %[[VAL_17:.*]] = hlfir.eval_in_mem shape %{{.*}} : (!fir.shape<1>) -> !hlfir.expr<10xf32> {
! CHECK:           ^bb0(%[[VAL_18:.*]]: !fir.ref<!fir.array<10xf32>>):
! CHECK:             %[[VAL_19:.*]] = fir.call @_QParray_func2(%[[VAL_5]], %[[VAL_6]]#0) fastmath<contract> : (!fir.ref<!fir.array<?xf32>>, !fir.ref<f32>) -> !fir.array<10xf32>
! CHECK:             fir.save_result %[[VAL_19]] to %[[VAL_18]](%{{.*}}) : !fir.array<10xf32>, !fir.ref<!fir.array<10xf32>>, !fir.shape<1>
! CHECK:           }
! CHECK:           hlfir.copy_out %{{.*}}, %[[VAL_4]]#1 to %{{.*}} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, i1, !fir.box<!fir.array<?xf32>>) -> ()
! CHECK:           hlfir.assign %[[VAL_17]] to %{{.*}} : !hlfir.expr<10xf32>, !fir.box<!fir.array<?xf32>>
! CHECK:           hlfir.end_associate %[[VAL_6]]#1, %[[VAL_6]]#2 : !fir.ref<f32>, i1
! CHECK:           hlfir.destroy %[[VAL_17]] : !hlfir.expr<10xf32>

subroutine test3(x)
  interface
     function array_func3(x)
       real :: x, array_func3(10)
     end function array_func3
  end interface
  logical :: x
  if (any(array_func3(1.0).le.array_func3(2.0))) x = .true.
end subroutine test3
! CHECK-LABEL:   func.func @_QPtest3(
! CHECK:           %[[VAL_2:.*]] = arith.constant 1.000000e+00 : f32
! CHECK:           %[[VAL_3:.*]]:3 = hlfir.associate %[[VAL_2]] {adapt.valuebyref} : (f32) -> (!fir.ref<f32>, !fir.ref<f32>, i1)
! CHECK:           %[[VAL_14:.*]] = hlfir.eval_in_mem shape %{{.*}} : (!fir.shape<1>) -> !hlfir.expr<10xf32> {
! CHECK:           ^bb0(%[[VAL_15:.*]]: !fir.ref<!fir.array<10xf32>>):
! CHECK:             %[[VAL_16:.*]] = fir.call @_QParray_func3(%[[VAL_3]]#0) fastmath<contract> : (!fir.ref<f32>) -> !fir.array<10xf32>
! CHECK:             fir.save_result %[[VAL_16]] to %[[VAL_15]](%{{.*}}) : !fir.array<10xf32>, !fir.ref<!fir.array<10xf32>>, !fir.shape<1>
! CHECK:           }
! CHECK:           %[[VAL_17:.*]] = arith.constant 2.000000e+00 : f32
! CHECK:           %[[VAL_18:.*]]:3 = hlfir.associate %[[VAL_17]] {adapt.valuebyref} : (f32) -> (!fir.ref<f32>, !fir.ref<f32>, i1)
! CHECK:           %[[VAL_29:.*]] = hlfir.eval_in_mem shape %{{.*}} : (!fir.shape<1>) -> !hlfir.expr<10xf32> {
! CHECK:           ^bb0(%[[VAL_30:.*]]: !fir.ref<!fir.array<10xf32>>):
! CHECK:             %[[VAL_31:.*]] = fir.call @_QParray_func3(%[[VAL_18]]#0) fastmath<contract> : (!fir.ref<f32>) -> !fir.array<10xf32>
! CHECK:             fir.save_result %[[VAL_31]] to %[[VAL_30]](%{{.*}}) : !fir.array<10xf32>, !fir.ref<!fir.array<10xf32>>, !fir.shape<1>
! CHECK:           }
! CHECK:           %[[VAL_32:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
! CHECK:           ^bb0(%[[VAL_33:.*]]: index):
! CHECK:             %[[VAL_34:.*]] = hlfir.apply %[[VAL_14]], %[[VAL_33]] : (!hlfir.expr<10xf32>, index) -> f32
! CHECK:             %[[VAL_35:.*]] = hlfir.apply %[[VAL_29]], %[[VAL_33]] : (!hlfir.expr<10xf32>, index) -> f32
! CHECK:             %[[VAL_36:.*]] = arith.cmpf ole, %[[VAL_34]], %[[VAL_35]] fastmath<contract> : f32
! CHECK:             %[[VAL_37:.*]] = fir.convert %[[VAL_36]] : (i1) -> !fir.logical<4>
! CHECK:             hlfir.yield_element %[[VAL_37]] : !fir.logical<4>
! CHECK:           }
! CHECK:           %[[VAL_38:.*]] = hlfir.any %[[VAL_32]] : (!hlfir.expr<?x!fir.logical<4>>) -> !fir.logical<4>
! CHECK:           hlfir.destroy %[[VAL_32]] : !hlfir.expr<?x!fir.logical<4>>
! CHECK:           hlfir.end_associate %[[VAL_18]]#1, %[[VAL_18]]#2 : !fir.ref<f32>, i1
! CHECK:           hlfir.destroy %[[VAL_29]] : !hlfir.expr<10xf32>
! CHECK:           hlfir.end_associate %[[VAL_3]]#1, %[[VAL_3]]#2 : !fir.ref<f32>, i1
! CHECK:           hlfir.destroy %[[VAL_14]] : !hlfir.expr<10xf32>
! CHECK:           %[[VAL_39:.*]] = fir.convert %[[VAL_38]] : (!fir.logical<4>) -> i1
! CHECK:           fir.if %[[VAL_39]] {

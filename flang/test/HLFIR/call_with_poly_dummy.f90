! RUN: bbc -polymorphic-type -emit-hlfir %s -o - | FileCheck %s

! Test passing arguments to subprograms with polymorphic dummy arguments.

! CHECK-LABEL:   func.func @_QPtest1() {
! CHECK:           %[[VAL_0:.*]] = arith.constant 17 : i32
! CHECK:           %[[VAL_1:.*]]:3 = hlfir.associate %[[VAL_0]] {uniq_name = "adapt.valuebyref"} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:           %[[VAL_2:.*]] = fir.embox %[[VAL_1]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK:           %[[VAL_3:.*]] = fir.rebox %[[VAL_2]] : (!fir.box<i32>) -> !fir.class<none>
! CHECK:           fir.call @_QPcallee(%[[VAL_3]]) fastmath<contract> : (!fir.class<none>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_1]]#1, %[[VAL_1]]#2 : !fir.ref<i32>, i1
! CHECK:           return
! CHECK:         }
subroutine test1
  interface
     subroutine callee(x)
       class(*) x
     end subroutine callee
  end interface
  call callee(17)
end subroutine test1

! CHECK-LABEL:   func.func @_QPtest2(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<f32> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtest2Ex"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<f32>
! CHECK:           %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:           %[[VAL_4:.*]] = arith.cmpf oeq, %[[VAL_2]], %[[VAL_3]] : f32
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i1) -> !fir.logical<4>
! CHECK:           %[[VAL_6:.*]]:3 = hlfir.associate %[[VAL_5]] {uniq_name = "adapt.valuebyref"} : (!fir.logical<4>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>, i1)
! CHECK:           %[[VAL_7:.*]] = fir.embox %[[VAL_6]]#0 : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
! CHECK:           %[[VAL_8:.*]] = fir.rebox %[[VAL_7]] : (!fir.box<!fir.logical<4>>) -> !fir.class<none>
! CHECK:           fir.call @_QPcallee(%[[VAL_8]]) fastmath<contract> : (!fir.class<none>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_6]]#1, %[[VAL_6]]#2 : !fir.ref<!fir.logical<4>>, i1
! CHECK:           return
! CHECK:         }
subroutine test2(x)
  interface
     subroutine callee(x)
       class(*) x
     end subroutine callee
  end interface
  call callee(x.eq.0)
end subroutine test2

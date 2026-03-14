! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s


! CHECK-LABEL:   func.func @_QPtest1(
! CHECK-SAME:        %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "x"},
! CHECK-SAME:        %[[VAL_1:.*]]: !fir.boxchar<1> {fir.bindc_name = "c"}) {
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.char<1>
! CHECK:           %[[VAL_D1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1>>
! CHECK:           %[[VAL_4_DECL:.*]]:2 = hlfir.declare %[[VAL_4]]
! CHECK:           %[[VAL_0_DECL:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_0_DECL]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_6_0:.*]] = fir.convert %[[VAL_5]] : (i32) -> i64
! CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_6_0]] : (i64) -> i8
! CHECK:           %[[VAL_7:.*]] = fir.undefined !fir.char<1>
! CHECK:           %[[VAL_8:.*]] = fir.insert_value %[[VAL_7]], %[[VAL_6]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
! CHECK:           fir.store %[[VAL_8]] to %[[VAL_2]] : !fir.ref<!fir.char<1>>
! CHECK:           %[[VAL_11:.*]] = hlfir.as_expr %[[VAL_2]] move %{{.*}} : (!fir.ref<!fir.char<1>>, i1) -> !hlfir.expr<!fir.char<1>>
! CHECK:           hlfir.assign %[[VAL_11]] to %[[VAL_4_DECL]]#0 : !hlfir.expr<!fir.char<1>>, !fir.ref<!fir.char<1>>
! CHECK:           hlfir.destroy %[[VAL_11]] : !hlfir.expr<!fir.char<1>>
! CHECK:           return
! CHECK:         }
subroutine test1(x, c)
  integer :: x
  character :: c
  c = achar(x)
end subroutine

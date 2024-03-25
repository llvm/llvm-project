! RUN: bbc -emit-fir -hlfir=false %s -o - | fir-opt --canonicalize | FileCheck %s
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir %s -o - | fir-opt --canonicalize | FileCheck %s


! CHECK-LABEL:   func.func @_QPtest1(
! CHECK-SAME:        %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "x"},
! CHECK-SAME:        %[[VAL_1:.*]]: !fir.boxchar<1> {fir.bindc_name = "c"}) {
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.char<1> {adapt.valuebyref}
! CHECK:           %[[VAL_3:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1>>
! CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i32) -> i8
! CHECK:           %[[VAL_7:.*]] = fir.undefined !fir.char<1>
! CHECK:           %[[VAL_8:.*]] = fir.insert_value %[[VAL_7]], %[[VAL_6]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
! CHECK:           fir.store %[[VAL_8]] to %[[VAL_2]] : !fir.ref<!fir.char<1>>
! CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.char<1>>
! CHECK:           fir.store %[[VAL_9]] to %[[VAL_4]] : !fir.ref<!fir.char<1>>
! CHECK:           return
! CHECK:         }
subroutine test1(x, c)
  integer :: x
  character :: c
  c = achar(x)
end subroutine

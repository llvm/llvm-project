! Test lowering of function return to HLFIR
! RUN: bbc -emit-fir -hlfir -o - %s 2>&1 | FileCheck %s

integer function simple_return()
  simple_return = 42
end function
! CHECK-LABEL: func.func @_QPsimple_return() -> i32 {
! CHECK:  %[[VAL_0:.*]] = fir.alloca i32
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFsimple_returnEsimple_return"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:  %[[VAL_2:.*]] = arith.constant 42 : i32
! CHECK:  hlfir.assign %[[VAL_2]] to %[[VAL_1]]#0 : i32, !fir.ref<i32>
! CHECK:  %[[VAL_3:.*]] = fir.load %[[VAL_1]]#1 : !fir.ref<i32>
! CHECK:  return %[[VAL_3]] : i32

character(10) function char_return()
  char_return = "hello"
end function
! CHECK-LABEL: func.func @_QPchar_return(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.char<1,10>>
! CHECK:  %[[VAL_2:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<!fir.char<1,?>>
! CHECK:  %[[VAL_3:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_2]] typeparams %[[VAL_3]] {uniq_name = "_QFchar_returnEchar_return"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:  %[[VAL_8:.*]] = fir.emboxchar %[[VAL_4]]#1, %[[VAL_3]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:  return %[[VAL_8]] : !fir.boxchar<1>

integer function array_return()
  dimension :: array_return(10)
  array_return = 42
end function
! CHECK-LABEL: func.func @_QParray_return() -> !fir.array<10xi32> {
! CHECK:  %[[VAL_1:.*]] = fir.alloca !fir.array<10xi32>
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]]{{.*}} {uniq_name = "_QFarray_returnEarray_return"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:  %[[VAL_4:.*]] = fir.load %[[VAL_3]]#1 : !fir.ref<!fir.array<10xi32>>
! CHECK:  return %[[VAL_4]] : !fir.array<10xi32>

character(5) function char_array_return()
  dimension :: char_array_return(10)
  char_array_return = "hello"
end function
! CHECK-LABEL: func.func @_QPchar_array_return() -> !fir.array<10x!fir.char<1,5>> {
! CHECK:  %[[VAL_2:.*]] = fir.alloca !fir.array<10x!fir.char<1,5>>
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_2]]{{.*}} {uniq_name = "_QFchar_array_returnEchar_array_return"} : (!fir.ref<!fir.array<10x!fir.char<1,5>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<10x!fir.char<1,5>>>, !fir.ref<!fir.array<10x!fir.char<1,5>>>)
! CHECK:  %[[VAL_5:.*]] = fir.load %[[VAL_4]]#1 : !fir.ref<!fir.array<10x!fir.char<1,5>>>
! CHECK:  return %[[VAL_5]] : !fir.array<10x!fir.char<1,5>>

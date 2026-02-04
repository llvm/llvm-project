! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: substring_main
subroutine substring_main
  character*7 :: string(2) = ['12     ', '12     ']
  integer :: result(2)
  integer :: ival
interface
  elemental function inner(arg)
    character(len=*), intent(in) :: arg
    integer :: inner
  end function inner
end interface

  ival = 1
  ! CHECK: %[[VAL_1:.*]] = fir.alloca i32 {bindc_name = "ival", uniq_name = "_QFsubstring_mainEival"}
  ! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFsubstring_mainEival"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK: %[[VAL_6:.*]] = fir.address_of(@_QFsubstring_mainEstring) : !fir.ref<!fir.array<2x!fir.char<1,7>>>
  ! CHECK: %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_6]](%{{.*}}) typeparams %{{.*}} {uniq_name = "_QFsubstring_mainEstring"} : (!fir.ref<!fir.array<2x!fir.char<1,7>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<2x!fir.char<1,7>>>, !fir.ref<!fir.array<2x!fir.char<1,7>>>)
  ! CHECK: %[[VAL_10:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<i32>
  ! CHECK: %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i32) -> i64
  ! CHECK: %[[VAL_12:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<i32>
  ! CHECK: %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i32) -> i64
  ! CHECK: %[[VAL_14:.*]] = fir.convert %[[VAL_11]] : (i64) -> index
  ! CHECK: %[[VAL_15:.*]] = fir.convert %[[VAL_13]] : (i64) -> index
  ! CHECK: %[[VAL_20:.*]] = hlfir.designate %[[VAL_8]]#0 (%{{.*}}:%{{.*}}:%{{.*}}) substr %[[VAL_14]], %[[VAL_15]]  shape %{{.*}} typeparams %{{.*}} : (!fir.ref<!fir.array<2x!fir.char<1,7>>>, index, index, index, index, index, !fir.shape<1>, index) -> !fir.box<!fir.array<2x!fir.char<1,?>>>
  ! CHECK: %[[VAL_21:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<2xi32> {
  ! CHECK: ^bb0(%[[VAL_22:.*]]: index):
  ! CHECK:   %[[VAL_23:.*]] = hlfir.designate %[[VAL_20]] (%[[VAL_22]])  typeparams %{{.*}} : (!fir.box<!fir.array<2x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
  ! CHECK:   %[[VAL_24:.*]] = fir.call @_QPinner(%[[VAL_23]]) {{.*}} : (!fir.boxchar<1>) -> i32
  ! CHECK:   hlfir.yield_element %[[VAL_24]] : i32
  ! CHECK: }
  result = inner(string(1:2)(ival:ival))
  print *, result
end subroutine substring_main

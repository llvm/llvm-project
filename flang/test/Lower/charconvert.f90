! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

subroutine test_c1_to_c4(c4, c1)
  character(len=*, kind=4) :: c4                           
  character(len=*, kind=1) :: c1            
  c4 = c1                                         
end subroutine                                           

subroutine test_c4_to_c1(c4, c1)
  character(len=*, kind=4) :: c4                           
  character(len=*, kind=1) :: c1            
  c1 = c4                                         
end subroutine                                           
                                                
! CHECK: func.func @_QPtest_c1_to_c4(%[[ARG0:.*]]: !fir.boxchar<4> {fir.bindc_name = "c4"}, %[[ARG1:.*]]: !fir.boxchar<1> {fir.bindc_name = "c1"}) {
! CHECK:   %[[VAL_0:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:   %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]]#0 typeparams %[[VAL_0]]#1 {uniq_name = "_QFtest_c1_to_c4Ec1"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:   %[[VAL_2:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<4>) -> (!fir.ref<!fir.char<4,?>>, index)
! CHECK:   %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]]#0 typeparams %[[VAL_2]]#1 {uniq_name = "_QFtest_c1_to_c4Ec4"} : (!fir.ref<!fir.char<4,?>>, index) -> (!fir.boxchar<4>, !fir.ref<!fir.char<4,?>>)
! CHECK:   %[[VAL_4:.*]] = fir.alloca !fir.char<4,?>(%[[VAL_0]]#1 : index)
! CHECK:   fir.char_convert %[[VAL_1]]#1 for %[[VAL_0]]#1 to %[[VAL_4:.*]] : !fir.ref<!fir.char<1,?>>, index, !fir.ref<!fir.char<4,?>>

! CHECK: func.func @_QPtest_c4_to_c1(%[[ARG0:.*]]: !fir.boxchar<4> {fir.bindc_name = "c4"}, %[[ARG1:.*]]: !fir.boxchar<1> {fir.bindc_name = "c1"}) {
! CHECK:   %[[VAL_0:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:   %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]]#0 typeparams %[[VAL_0]]#1 {uniq_name = "_QFtest_c4_to_c1Ec1"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:   %[[VAL_2:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<4>) -> (!fir.ref<!fir.char<4,?>>, index)
! CHECK:   %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]]#0 typeparams %[[VAL_2]]#1 {uniq_name = "_QFtest_c4_to_c1Ec4"} : (!fir.ref<!fir.char<4,?>>, index) -> (!fir.boxchar<4>, !fir.ref<!fir.char<4,?>>)
! CHECK:   %[[C4:.*]] = arith.constant 4 : index
! CHECK:   %[[VAL_4:.*]] = arith.muli %[[VAL_2]]#1, %[[C4]] : index
! CHECK:   %[[VAL_5:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_4]] : index)
! CHECK:   fir.char_convert %[[VAL_3]]#1 for %[[VAL_2]]#1 to %[[VAL_5:.*]] : !fir.ref<!fir.char<4,?>>, index, !fir.ref<!fir.char<1,?>>
! CHECK:   %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] typeparams %[[VAL_2]]#1 {uniq_name = "ctor.temp"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
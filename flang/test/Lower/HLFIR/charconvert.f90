! Test lowering of character concatenation to HLFIR
! RUN: bbc -emit-hlfir -o - %s 2>&1 | FileCheck %s

subroutine charconvert1(c,n)
  character(*,4),intent(in) :: c(:)
  integer,intent(in) :: n
  interface
     subroutine callee(c)
       character(*),intent(in) :: c(:)
     end subroutine callee
  end interface
  call show([character(n)::c])
end subroutine charconvert1

! CHECK-LABEL: func.func @_QPcharconvert1
! CHECK:   %[[VAL_2:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFcharconvert1Ec"} : (!fir.box<!fir.array<?x!fir.char<4,?>>>) -> (!fir.box<!fir.array<?x!fir.char<4,?>>>, !fir.box<!fir.array<?x!fir.char<4,?>>>)
! CHECK:   ^bb0(%[[ARG2:.*]]: index):
! CHECK:     %[[VAL_37:.*]] = fir.box_elesize %[[VAL_2]]#1 : (!fir.box<!fir.array<?x!fir.char<4,?>>>) -> index
! CHECK:     %[[C4_4:.*]] = arith.constant 4 : index
! CHECK:     %[[VAL_38:.*]] = arith.divsi %[[VAL_37]], %[[C4_4]] : index
! CHECK:     %[[VAL_39:.*]] = hlfir.designate %[[VAL_2]]#0 (%[[ARG2]])  typeparams %[[VAL_38]] : (!fir.box<!fir.array<?x!fir.char<4,?>>>, index, index) -> !fir.boxchar<4>
! CHECK:     %[[VAL_42:.*]]:2 = fir.unboxchar %[[VAL_39]] : (!fir.boxchar<4>) -> (!fir.ref<!fir.char<4,?>>, index)
! CHECK:     %[[C4_5:.*]] = arith.constant 4 : index
! CHECK:     %[[VAL_40:.*]] = arith.muli %[[VAL_38]], %[[C4_5]] : index
! CHECK:     %[[VAL_41:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_40]] : index)
! CHECK:     fir.char_convert %[[VAL_42]]#0 for %[[VAL_38:.*]] to %[[VAL_41]] : !fir.ref<!fir.char<4,?>>, index, !fir.ref<!fir.char<1,?>>

subroutine charconvert2(x)
  integer,intent(in) :: x
  character(kind=4) :: cx
  cx = achar(x)
end subroutine charconvert2
! CHECK-LABEL: func.func @_QPcharconvert2
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<i32>
! CHECK:   %[[VAL_0:.*]] = fir.alloca !fir.char<1>
! CHECK:   %[[C1:.*]] = arith.constant 1 : index
! CHECK:   %[[VAL_1:.*]] = fir.alloca !fir.char<4> {bindc_name = "cx", uniq_name = "_QFcharconvert2Ecx"}
! CHECK:   %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] typeparams %[[C1]] {uniq_name = "_QFcharconvert2Ecx"} : (!fir.ref<!fir.char<4>>, index) -> (!fir.ref<!fir.char<4>>, !fir.ref<!fir.char<4>>)
! CHECK:   %[[VAL_3:.*]]:2 = hlfir.declare %[[ARG0]] {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFcharconvert2Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:   %[[VAL_4:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i32>
! CHECK:   %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> i64
! CHECK:   %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i64) -> i8
! CHECK:   %[[VAL_7:.*]] = fir.undefined !fir.char<1>
! CHECK:   %[[VAL_8:.*]] = fir.insert_value %[[VAL_7]], %[[VAL_6]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
! CHECK:   fir.store %[[VAL_8:.*]] to %[[VAL_0]] : !fir.ref<!fir.char<1>>
! CHECK:   %[[FALSE:.*]] = arith.constant false
! CHECK:   %[[VAL_9:.*]] = hlfir.as_expr %[[VAL_0]] move %[[FALSE]] : (!fir.ref<!fir.char<1>>, i1) -> !hlfir.expr<!fir.char<1>>
! CHECK:   %[[C1_0:.*]] = arith.constant 1 : index
! CHECK:   %[[VAL_10:.*]] = fir.alloca !fir.char<4,?>(%[[C1_0]] : index)
! CHECK:   fir.char_convert %[[VAL_0:.*]] for %[[C1_0]] to %[[VAL_10]] : !fir.ref<!fir.char<1>>, index, !fir.ref<!fir.char<4,?>>

subroutine charconvert3(c, c4)
  character(kind=1, len=*) :: c
  character(kind=4, len=*) :: c4
  c4 = c // c
end subroutine

! CHECK-LABEL: func.func @_QPcharconvert3
! CHECK-SAME: %[[ARG0:.*]]: !fir.boxchar<1> {{.*}}, %[[ARG1:.*]]: !fir.boxchar<4> 
! CHECK:   %[[VAL_0:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:   %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]]#0 typeparams %[[VAL_0]]#1 {uniq_name = "_QFcharconvert3Ec"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:   %[[VAL_2:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<4>) -> (!fir.ref<!fir.char<4,?>>, index)
! CHECK:   %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]]#0 typeparams %[[VAL_2]]#1 {uniq_name = "_QFcharconvert3Ec4"} : (!fir.ref<!fir.char<4,?>>, index) -> (!fir.boxchar<4>, !fir.ref<!fir.char<4,?>>)
! CHECK:   %[[VAL_4:.*]] = arith.addi %[[VAL_0]]#1, %[[VAL_0]]#1 : index
! CHECK:   %[[VAL_5:.*]] = hlfir.concat %[[VAL_1]]#0, %[[VAL_1]]#0 len %[[VAL_4]] : (!fir.boxchar<1>, !fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,?>>
! CHECK:   %[[VAL_7:.*]]:3 = hlfir.associate %[[VAL_5]] typeparams %[[VAL_4]] {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
! CHECK:   %[[VAL_6:.*]] = fir.alloca !fir.char<4,?>(%[[VAL_4]] : index)
! CHECK:   fir.char_convert %[[VAL_7]]#1 for %[[VAL_4:.*]] to %[[VAL_6]] : !fir.ref<!fir.char<1,?>>, index, !fir.ref<!fir.char<4,?>>
! CHECK:   hlfir.end_associate %[[VAL_7]]#1, %[[VAL_7]]#2 : !fir.ref<!fir.char<1,?>>, i1
! CHECK:   %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_6]] typeparams %[[VAL_4]] {uniq_name = ".temp.kindconvert"} : (!fir.ref<!fir.char<4,?>>, index) -> (!fir.boxchar<4>, !fir.ref<!fir.char<4,?>>)
! CHECK:   hlfir.assign %[[VAL_8]]#0 to %[[VAL_3]]#0 : !fir.boxchar<4>, !fir.boxchar<4>

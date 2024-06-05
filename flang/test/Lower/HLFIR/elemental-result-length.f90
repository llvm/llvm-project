! RUN: bbc -emit-hlfir -o - %s | fir-opt --canonicalize | FileCheck %s

module m1
contains
elemental function fct1(a, b) result(t)
  character(*), intent(in) :: a, b
  character(len(a) + len(b)) :: t
  t = a // b
end function

subroutine sub2(a,b,c)
  character(*), intent(inout) :: c
  character(*), intent(in) :: a, b

  c = fct1(a,b)
end subroutine

! CHECK-LABEL: func.func @_QMm1Psub2(
! CHECK-SAME: %[[ARG0:.*]]: !fir.boxchar<1> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.boxchar<1> {fir.bindc_name = "b"}, %[[ARG2:.*]]: !fir.boxchar<1> {fir.bindc_name = "c"}) {
! CHECK: %[[UNBOX_ARG0:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[A:.*]]:2 = hlfir.declare %[[UNBOX_ARG0]]#0 typeparams %[[UNBOX_ARG0]]#1 dummy_scope %0 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QMm1Fsub2Ea"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK: %[[UNBOX_ARG1:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[B:.*]]:2 = hlfir.declare %[[UNBOX_ARG1]]#0 typeparams %[[UNBOX_ARG1]]#1 dummy_scope %{{.*}} {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QMm1Fsub2Eb"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK: %[[UNBOX_ARG2:.*]]:2 = fir.unboxchar %[[ARG2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[C:.*]]:2 = hlfir.declare %[[UNBOX_ARG2]]#0 typeparams %[[UNBOX_ARG2]]#1 dummy_scope %{{.*}} {fortran_attrs = #fir.var_attrs<intent_inout>, uniq_name = "_QMm1Fsub2Ec"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK: %[[UNBOX_A:.*]]:2 = fir.unboxchar %[[A]]#0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[DUMMYA:.*]]:2 = hlfir.declare %[[UNBOX_A]]#0 typeparams %[[UNBOX_A]]#1 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QMm1Ffct1Ea"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK: %[[UNBOX_B:.*]]:2 = fir.unboxchar %[[B]]#0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[DUMMYB:.*]]:2 = hlfir.declare %[[UNBOX_B]]#0 typeparams %[[UNBOX_B]]#1 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QMm1Ffct1Eb"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK: %[[LEN_A:.*]] = fir.convert %[[UNBOX_A]]#1 : (index) -> i32
! CHECK: %[[LEN_B:.*]] = fir.convert %[[UNBOX_B]]#1 : (index) -> i32
! CHECK: %[[LEN_LEN:.*]] = arith.addi %[[LEN_A]], %[[LEN_B]] : i32
! CHECK: %[[LEN_LEN_IDX:.*]] = fir.convert %[[LEN_LEN]] : (i32) -> index
! CHECK: %[[CMPI:.*]] = arith.cmpi sgt, %[[LEN_LEN_IDX]], %c0{{.*}} : index
! CHECK: %[[RES_LENGTH:.*]] = arith.select %[[CMPI]], %[[LEN_LEN_IDX]], %c0{{.*}} : index
! CHECK: %[[RES:.*]] = fir.alloca !fir.char<1,?>(%[[RES_LENGTH]] : index) {bindc_name = ".result"}
! CHECK: fir.call @_QMm1Pfct1

subroutine sub4(a,b,c)
  character(*), intent(inout) :: c(:)
  character(*), intent(in) :: a(:), b(:)

  c = fct1(a,b)
end subroutine

! CHECK-LABEL: func.func @_QMm1Psub4(
! CHECK-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>> {fir.bindc_name = "b"}, %[[ARG2:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>> {fir.bindc_name = "c"}) {
! CHECK: %[[A:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{.*}} {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QMm1Fsub4Ea"} : (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.box<!fir.array<?x!fir.char<1,?>>>)
! CHECK: %[[B:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %{{.*}} {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QMm1Fsub4Eb"} : (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.box<!fir.array<?x!fir.char<1,?>>>)
! CHECK: %[[C:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %{{.*}} {fortran_attrs = #fir.var_attrs<intent_inout>, uniq_name = "_QMm1Fsub4Ec"} : (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.box<!fir.array<?x!fir.char<1,?>>>)
! CHECK: %[[LEN_A:.*]] = fir.box_elesize %[[A]]#1 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK: %[[LEN_B:.*]] = fir.box_elesize %[[B]]#1 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK: %[[LEN_A_I32:.*]] = fir.convert %[[LEN_A]] : (index) -> i32
! CHECK: %[[LEN_B_I32:.*]] = fir.convert %[[LEN_B]] : (index) -> i32
! CHECK: %[[LEN_LEN:.*]] = arith.addi %[[LEN_A_I32]], %[[LEN_B_I32]] : i32
! CHECK: %[[LEN_LEN_IDX:.*]] = fir.convert %[[LEN_LEN]] : (i32) -> index
! CHECK: %[[CMPI:.*]] = arith.cmpi sgt, %[[LEN_LEN_IDX]], %c0{{.*}} : index
! CHECK: %[[LENGTH:.*]] = arith.select %[[CMPI]], %17, %c0{{.*}} : index
! CHECK: %{{.*}} = hlfir.elemental %{{.*}} typeparams %[[LENGTH]] unordered : (!fir.shape<1>, index) -> !hlfir.expr<?x!fir.char<1,?>>

end module

program test
  use m1
  character(5) :: a(2) = ['abcde', 'klmnop'], b(2) = ['fghij', 'qrstu']
  character(10) :: c(2)

  call sub2(a(1), b(1), c(1))
  print*, c(1)
end

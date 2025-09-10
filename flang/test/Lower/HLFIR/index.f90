! Test lowering of index intrinsic to HLFIR
! RUN: bbc -emit-hlfir -o - %s 2>&1 | FileCheck %s

subroutine t(s)
  implicit none
  character(len=*, kind=1):: s
  integer :: n
  n = index(s,'this')
end subroutine t
! CHECK-LABEL:   func.func @_QPt(
! CHECK-SAME:                    %[[ARG0:.*]]: !fir.boxchar<1> {fir.bindc_name = "s"}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]] = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFtEn"}
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFtEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_3:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]]#0 typeparams %[[VAL_3]]#1 dummy_scope %[[VAL_0]] {uniq_name = "_QFtEs"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:           %[[VAL_5:.*]] = fir.address_of(@_QQclX74686973) : !fir.ref<!fir.char<1,4>>
! CHECK:           %[[VAL_6:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_5]] typeparams %[[VAL_6]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX74686973"} : (!fir.ref<!fir.char<1,4>>, index) -> (!fir.ref<!fir.char<1,4>>, !fir.ref<!fir.char<1,4>>)
! CHECK:           %[[VAL_8:.*]] = hlfir.index %[[VAL_7]]#0 in %[[VAL_4]]#0 : (!fir.ref<!fir.char<1,4>>, !fir.boxchar<1>) -> i32
! CHECK:           hlfir.assign %[[VAL_8]] to %[[VAL_2]]#0 : i32, !fir.ref<i32>

subroutine t1(s, b)
  implicit none
  character(len=*, kind=1):: s
  logical :: b
  integer :: n
  n = index(s,'this', back = b)
end subroutine t1
! CHECK-LABEL:   func.func @_QPt1(
! CHECK-SAME:                     %[[ARG0:.*]]: !fir.boxchar<1> {fir.bindc_name = "s"},
! CHECK-SAME:                     %[[ARG1:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "b"}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_0]] {uniq_name = "_QFt1Eb"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:           %[[VAL_2:.*]] = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFt1En"}
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFt1En"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_4:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]]#0 typeparams %[[VAL_4]]#1 dummy_scope %[[VAL_0]] {uniq_name = "_QFt1Es"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:           %[[VAL_6:.*]] = fir.address_of(@_QQclX74686973) : !fir.ref<!fir.char<1,4>>
! CHECK:           %[[VAL_7:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_6]] typeparams %[[VAL_7]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX74686973"} : (!fir.ref<!fir.char<1,4>>, index) -> (!fir.ref<!fir.char<1,4>>, !fir.ref<!fir.char<1,4>>)
! CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:           %[[VAL_10:.*]] = hlfir.index %[[VAL_8]]#0 in %[[VAL_5]]#0 back %[[VAL_9]] : (!fir.ref<!fir.char<1,4>>, !fir.boxchar<1>, !fir.logical<4>) -> i32
! CHECK:           hlfir.assign %[[VAL_10]] to %[[VAL_3]]#0 : i32, !fir.ref<i32>


subroutine t2(s, c)
  implicit none
  character(len=*, kind=2):: s, c
  integer :: n
  n = index(s,c,back=.false.)
end subroutine t2
! CHECK-LABEL:   func.func @_QPt2(
! CHECK-SAME:                     %[[ARG0:.*]]: !fir.boxchar<2> {fir.bindc_name = "s"},
! CHECK-SAME:                     %[[ARG1:.*]]: !fir.boxchar<2> {fir.bindc_name = "c"}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]]#0 typeparams %[[VAL_1]]#1 dummy_scope %[[VAL_0]] {uniq_name = "_QFt2Ec"} : (!fir.ref<!fir.char<2,?>>, index, !fir.dscope) -> (!fir.boxchar<2>, !fir.ref<!fir.char<2,?>>)
! CHECK:           %[[VAL_3:.*]] = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFt2En"}
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] {uniq_name = "_QFt2En"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_5:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]]#0 typeparams %[[VAL_5]]#1 dummy_scope %[[VAL_0]] {uniq_name = "_QFt2Es"} : (!fir.ref<!fir.char<2,?>>, index, !fir.dscope) -> (!fir.boxchar<2>, !fir.ref<!fir.char<2,?>>)
! CHECK:           %[[VAL_7:.*]] = arith.constant false
! CHECK:           %[[VAL_8:.*]] = hlfir.index %[[VAL_2]]#0 in %[[VAL_6]]#0 back %[[VAL_7]] : (!fir.boxchar<2>, !fir.boxchar<2>, i1) -> i32
! CHECK:           hlfir.assign %[[VAL_8]] to %[[VAL_4]]#0 : i32, !fir.ref<i32>

subroutine t3(s, c)
  implicit none
  character(len=*, kind=4):: s, c
  integer :: n
  n = index(s,c,back=.true., kind=1)
end subroutine t3
! CHECK-LABEL:   func.func @_QPt3(
! CHECK-SAME:                     %[[ARG0:.*]]: !fir.boxchar<4> {fir.bindc_name = "s"},
! CHECK-SAME:                     %[[ARG1:.*]]: !fir.boxchar<4> {fir.bindc_name = "c"}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<4>) -> (!fir.ref<!fir.char<4,?>>, index)
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]]#0 typeparams %[[VAL_1]]#1 dummy_scope %[[VAL_0]] {uniq_name = "_QFt3Ec"} : (!fir.ref<!fir.char<4,?>>, index, !fir.dscope) -> (!fir.boxchar<4>, !fir.ref<!fir.char<4,?>>)
! CHECK:           %[[VAL_3:.*]] = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFt3En"}
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] {uniq_name = "_QFt3En"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_5:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<4>) -> (!fir.ref<!fir.char<4,?>>, index)
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]]#0 typeparams %[[VAL_5]]#1 dummy_scope %[[VAL_0]] {uniq_name = "_QFt3Es"} : (!fir.ref<!fir.char<4,?>>, index, !fir.dscope) -> (!fir.boxchar<4>, !fir.ref<!fir.char<4,?>>)
! CHECK:           %[[VAL_7:.*]] = arith.constant true
! CHECK:           %[[VAL_8:.*]] = hlfir.index %[[VAL_2]]#0 in %[[VAL_6]]#0 back %[[VAL_7]] : (!fir.boxchar<4>, !fir.boxchar<4>, i1) -> i8
! CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i8) -> i32
! CHECK:           hlfir.assign %[[VAL_9]] to %[[VAL_4]]#0 : i32, !fir.ref<i32>

subroutine t4(c1, c2)
  implicit none
  character(*) :: c1(3)
  character(*) :: c2(3)
  integer(kind=1) :: n(3)
  n = index(c1, c2, kind=1)
end subroutine t4
! CHECK-LABEL:   func.func @_QPt4(
! CHECK-SAME:                     %[[ARG0:.*]]: !fir.boxchar<1> {fir.bindc_name = "c1"},
! CHECK-SAME:                     %[[ARG1:.*]]: !fir.boxchar<1> {fir.bindc_name = "c2"}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_2:.*]] = fir.convert %[[VAL_1]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<3x!fir.char<1,?>>>
! CHECK:           %[[VAL_3:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_2]](%[[VAL_4]]) typeparams %[[VAL_1]]#1 dummy_scope %[[VAL_0]] {uniq_name = "_QFt4Ec1"} : (!fir.ref<!fir.array<3x!fir.char<1,?>>>, !fir.shape<1>, index, !fir.dscope) -> (!fir.box<!fir.array<3x!fir.char<1,?>>>, !fir.ref<!fir.array<3x!fir.char<1,?>>>)
! CHECK:           %[[VAL_6:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<3x!fir.char<1,?>>>
! CHECK:           %[[VAL_8:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_9:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_7]](%[[VAL_9]]) typeparams %[[VAL_6]]#1 dummy_scope %[[VAL_0]] {uniq_name = "_QFt4Ec2"} : (!fir.ref<!fir.array<3x!fir.char<1,?>>>, !fir.shape<1>, index, !fir.dscope) -> (!fir.box<!fir.array<3x!fir.char<1,?>>>, !fir.ref<!fir.array<3x!fir.char<1,?>>>)
! CHECK:           %[[VAL_11:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_12:.*]] = fir.alloca !fir.array<3xi8> {bindc_name = "n", uniq_name = "_QFt4En"}
! CHECK:           %[[VAL_13:.*]] = fir.shape %[[VAL_11]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_12]](%[[VAL_13]]) {uniq_name = "_QFt4En"} : (!fir.ref<!fir.array<3xi8>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi8>>, !fir.ref<!fir.array<3xi8>>)
! CHECK:           %[[VAL_15:.*]] = hlfir.elemental %[[VAL_4]] unordered : (!fir.shape<1>) -> !hlfir.expr<3xi8> {
! CHECK:           ^bb0(%[[VAL_16:.*]]: index):
! CHECK:             %[[VAL_17:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_16]])  typeparams %[[VAL_1]]#1 : (!fir.box<!fir.array<3x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
! CHECK:             %[[VAL_18:.*]] = hlfir.designate %[[VAL_10]]#0 (%[[VAL_16]])  typeparams %[[VAL_6]]#1 : (!fir.box<!fir.array<3x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
! CHECK:             %[[VAL_19:.*]] = hlfir.index %[[VAL_18]] in %[[VAL_17]] : (!fir.boxchar<1>, !fir.boxchar<1>) -> i8
! CHECK:             hlfir.yield_element %[[VAL_19]] : i8
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_15]] to %[[VAL_14]]#0 : !hlfir.expr<3xi8>, !fir.ref<!fir.array<3xi8>>

! index is called as elemental with the 3d argument optional for 'sub' (^bb0 block)
! Make sure that the argument is actually accessed (hlfir.designate) only
! under fir.if that depends on fir.is_present check.
program test
  call sub('abcdefgc',(/'c','c'/))
contains
  subroutine sub(a,b,c)
    character(*) a,b(:)
    logical,optional :: c(:)
    print *,index(a,b,c)
  end subroutine
end program test
! CHECK-LABEL:   func.func private @_QFPsub(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.boxchar<1> {fir.bindc_name = "a"},
! CHECK-SAME:      %[[ARG1:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>> {fir.bindc_name = "b"},
! CHECK-SAME:      %[[ARG2:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "c", fir.optional}) attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]]#0 typeparams %[[VAL_1]]#1 dummy_scope %[[VAL_0]] {uniq_name = "_QFFsubEa"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_0]] {uniq_name = "_QFFsubEb"} : (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.box<!fir.array<?x!fir.char<1,?>>>)
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %[[VAL_0]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFFsubEc"} : (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>)
! CHECK:           %[[VAL_10:.*]] = fir.is_present %[[VAL_4]]#0 : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> i1
! CHECK:           %[[VAL_11:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_12:.*]]:3 = fir.box_dims %[[VAL_3]]#0, %[[VAL_11]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_13:.*]] = fir.shape %[[VAL_12]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_14:.*]] = hlfir.elemental %[[VAL_13]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
! CHECK:           ^bb0(%[[VAL_15:.*]]: index):
! CHECK:             %[[VAL_16:.*]] = fir.box_elesize %[[VAL_3]]#1 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK:             %[[VAL_17:.*]] = hlfir.designate %[[VAL_3]]#0 (%[[VAL_15]])  typeparams %[[VAL_16]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
! CHECK:             %[[VAL_18:.*]] = fir.if %[[VAL_10]] -> (!fir.logical<4>) {
! CHECK:               %[[VAL_19:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_15]])  : (!fir.box<!fir.array<?x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
! CHECK:               %[[VAL_20:.*]] = fir.load %[[VAL_19]] : !fir.ref<!fir.logical<4>>
! CHECK:               fir.result %[[VAL_20]] : !fir.logical<4>
! CHECK:             } else {
! CHECK:               %[[VAL_21:.*]] = arith.constant false
! CHECK:               %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (i1) -> !fir.logical<4>
! CHECK:               fir.result %[[VAL_22]] : !fir.logical<4>
! CHECK:             }
! CHECK:             %[[VAL_23:.*]] = hlfir.index %[[VAL_17]] in %[[VAL_2]]#0 back %[[VAL_18]] : (!fir.boxchar<1>, !fir.boxchar<1>, !fir.logical<4>) -> i32
! CHECK:             hlfir.yield_element %[[VAL_23]] : i32
! CHECK:           }

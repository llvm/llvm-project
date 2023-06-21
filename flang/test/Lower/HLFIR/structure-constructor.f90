! Test lowering of StructureConstructor.
! RUN: bbc -emit-hlfir -o - -I nowhere %s 2>&1 | FileCheck %s

module types
  type t1
     character(4) :: c
  end type t1
  type t2
     integer :: i(10)
  end type t2
  type t3
     real, pointer :: r(:)
  end type t3
  type t4
     character(2), allocatable :: c(:)
  end type t4
  type t5
     type(t4), allocatable :: t5m(:)
  end type t5
  type, extends(t5) :: t6
     type(t1) :: t6m(1)
  end type t6
end module types

subroutine test1(x)
  use types
  character(4) :: x
  type(t1) :: res
  res = t1(x)
end subroutine test1
! CHECK-LABEL:   func.func @_QPtest1(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.boxchar<1> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.type<_QMtypesTt1{c:!fir.char<1,4>}>
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.type<_QMtypesTt1{c:!fir.char<1,4>}> {bindc_name = "res", uniq_name = "_QFtest1Eres"}
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFtest1Eres"} : (!fir.ref<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>) -> (!fir.ref<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>, !fir.ref<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>)
! CHECK:           %[[VAL_4:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_5:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_4]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,4>>
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]] typeparams %[[VAL_5]] {uniq_name = "_QFtest1Ex"} : (!fir.ref<!fir.char<1,4>>, index) -> (!fir.ref<!fir.char<1,4>>, !fir.ref<!fir.char<1,4>>)
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "ctor.temp"} : (!fir.ref<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>) -> (!fir.ref<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>, !fir.ref<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>)
! CHECK:           %[[VAL_9:.*]] = fir.embox %[[VAL_8]]#0 : (!fir.ref<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>) -> !fir.box<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>
! CHECK:           %[[VAL_10:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]*}}>>
! CHECK:           %[[VAL_11:.*]] = arith.constant {{[0-9]*}} : i32
! CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_9]] : (!fir.box<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>) -> !fir.box<none>
! CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_10]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_14:.*]] = fir.call @_FortranAInitialize(%[[VAL_12]], %[[VAL_13]], %[[VAL_11]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:           %[[VAL_15:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_16:.*]] = hlfir.designate %[[VAL_8]]#0{"c"}   typeparams %[[VAL_15]] : (!fir.ref<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>, index) -> !fir.ref<!fir.char<1,4>>
! CHECK:           %[[VAL_17:.*]] = arith.constant 4 : i64
! CHECK:           %[[VAL_18:.*]] = hlfir.set_length %[[VAL_7]]#0 len %[[VAL_17]] : (!fir.ref<!fir.char<1,4>>, i64) -> !hlfir.expr<!fir.char<1,4>>
! CHECK:           hlfir.assign %[[VAL_18]] to %[[VAL_16]] : !hlfir.expr<!fir.char<1,4>>, !fir.ref<!fir.char<1,4>>
! CHECK:           hlfir.assign %[[VAL_8]]#0 to %[[VAL_3]]#0 : !fir.ref<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>, !fir.ref<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>
! CHECK:           return
! CHECK:         }

subroutine test2(x)
  use types
  integer :: x(10)
  type(t2) res
  res = t2(x)
end subroutine test2
! CHECK-LABEL:   func.func @_QPtest2(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.type<_QMtypesTt2{i:!fir.array<10xi32>}>
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.type<_QMtypesTt2{i:!fir.array<10xi32>}> {bindc_name = "res", uniq_name = "_QFtest2Eres"}
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFtest2Eres"} : (!fir.ref<!fir.type<_QMtypesTt2{i:!fir.array<10xi32>}>>) -> (!fir.ref<!fir.type<_QMtypesTt2{i:!fir.array<10xi32>}>>, !fir.ref<!fir.type<_QMtypesTt2{i:!fir.array<10xi32>}>>)
! CHECK:           %[[VAL_4:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_5]]) {uniq_name = "_QFtest2Ex"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "ctor.temp"} : (!fir.ref<!fir.type<_QMtypesTt2{i:!fir.array<10xi32>}>>) -> (!fir.ref<!fir.type<_QMtypesTt2{i:!fir.array<10xi32>}>>, !fir.ref<!fir.type<_QMtypesTt2{i:!fir.array<10xi32>}>>)
! CHECK:           %[[VAL_8:.*]] = fir.embox %[[VAL_7]]#0 : (!fir.ref<!fir.type<_QMtypesTt2{i:!fir.array<10xi32>}>>) -> !fir.box<!fir.type<_QMtypesTt2{i:!fir.array<10xi32>}>>
! CHECK:           %[[VAL_9:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]*}}>>
! CHECK:           %[[VAL_10:.*]] = arith.constant {{[0-9]*}} : i32
! CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_8]] : (!fir.box<!fir.type<_QMtypesTt2{i:!fir.array<10xi32>}>>) -> !fir.box<none>
! CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_9]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_13:.*]] = fir.call @_FortranAInitialize(%[[VAL_11]], %[[VAL_12]], %[[VAL_10]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:           %[[VAL_14:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_15:.*]] = fir.shape %[[VAL_14]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_16:.*]] = hlfir.designate %[[VAL_7]]#0{"i"} <%[[VAL_15]]>   shape %[[VAL_15]] : (!fir.ref<!fir.type<_QMtypesTt2{i:!fir.array<10xi32>}>>, !fir.shape<1>, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>>
! CHECK:           hlfir.assign %[[VAL_6]]#0 to %[[VAL_16]] : !fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>
! CHECK:           hlfir.assign %[[VAL_7]]#0 to %[[VAL_3]]#0 : !fir.ref<!fir.type<_QMtypesTt2{i:!fir.array<10xi32>}>>, !fir.ref<!fir.type<_QMtypesTt2{i:!fir.array<10xi32>}>>
! CHECK:           return
! CHECK:         }

subroutine test3(x)
  use types
  real, pointer :: x(:)
  type(t3) res
  res = t3(x)
end subroutine test3
! CHECK-LABEL:   func.func @_QPtest3(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.type<_QMtypesTt3{r:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.type<_QMtypesTt3{r:!fir.box<!fir.ptr<!fir.array<?xf32>>>}> {bindc_name = "res", uniq_name = "_QFtest3Eres"}
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFtest3Eres"} : (!fir.ref<!fir.type<_QMtypesTt3{r:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>) -> (!fir.ref<!fir.type<_QMtypesTt3{r:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>, !fir.ref<!fir.type<_QMtypesTt3{r:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>)
! CHECK:           %[[VAL_4:.*]] = fir.embox %[[VAL_3]]#1 : (!fir.ref<!fir.type<_QMtypesTt3{r:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>) -> !fir.box<!fir.type<_QMtypesTt3{r:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>
! CHECK:           %[[VAL_5:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]*}}>>
! CHECK:           %[[VAL_6:.*]] = arith.constant {{[0-9]*}} : i32
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_4]] : (!fir.box<!fir.type<_QMtypesTt3{r:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>) -> !fir.box<none>
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_5]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_9:.*]] = fir.call @_FortranAInitialize(%[[VAL_7]], %[[VAL_8]], %[[VAL_6]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest3Ex"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "ctor.temp"} : (!fir.ref<!fir.type<_QMtypesTt3{r:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>) -> (!fir.ref<!fir.type<_QMtypesTt3{r:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>, !fir.ref<!fir.type<_QMtypesTt3{r:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>)
! CHECK:           %[[VAL_12:.*]] = fir.embox %[[VAL_11]]#0 : (!fir.ref<!fir.type<_QMtypesTt3{r:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>) -> !fir.box<!fir.type<_QMtypesTt3{r:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>
! CHECK:           %[[VAL_13:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]*}}>>
! CHECK:           %[[VAL_14:.*]] = arith.constant {{[0-9]*}} : i32
! CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_12]] : (!fir.box<!fir.type<_QMtypesTt3{r:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>) -> !fir.box<none>
! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_13]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_17:.*]] = fir.call @_FortranAInitialize(%[[VAL_15]], %[[VAL_16]], %[[VAL_14]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:           %[[VAL_18:.*]] = hlfir.designate %[[VAL_11]]#0{"r"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMtypesTt3{r:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:           %[[VAL_19:.*]] = fir.load %[[VAL_10]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:           %[[VAL_20:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_21:.*]]:3 = fir.box_dims %[[VAL_19]], %[[VAL_20]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_22:.*]] = fir.shift %[[VAL_21]]#0 : (index) -> !fir.shift<1>
! CHECK:           %[[VAL_23:.*]] = fir.rebox %[[VAL_19]](%[[VAL_22]]) : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.shift<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:           fir.store %[[VAL_23]] to %[[VAL_18]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:           hlfir.assign %[[VAL_11]]#0 to %[[VAL_3]]#0 : !fir.ref<!fir.type<_QMtypesTt3{r:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>, !fir.ref<!fir.type<_QMtypesTt3{r:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>
! CHECK:           return
! CHECK:         }

subroutine test4(x)
  use types
  character(2), allocatable :: x(:)
  type(t4) res
  res = t4(x)
end subroutine test4
! CHECK-LABEL:   func.func @_QPtest4(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}> {bindc_name = "res", uniq_name = "_QFtest4Eres"}
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFtest4Eres"} : (!fir.ref<!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>) -> (!fir.ref<!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>, !fir.ref<!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>)
! CHECK:           %[[VAL_4:.*]] = fir.embox %[[VAL_3]]#1 : (!fir.ref<!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>) -> !fir.box<!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>
! CHECK:           %[[VAL_5:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]*}}>>
! CHECK:           %[[VAL_6:.*]] = arith.constant {{[0-9]*}} : i32
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_4]] : (!fir.box<!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>) -> !fir.box<none>
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_5]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_9:.*]] = fir.call @_FortranAInitialize(%[[VAL_7]], %[[VAL_8]], %[[VAL_6]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:           %[[VAL_10:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[VAL_10]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest4Ex"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>>, index) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>>)
! CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "ctor.temp"} : (!fir.ref<!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>) -> (!fir.ref<!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>, !fir.ref<!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>)
! CHECK:           %[[VAL_13:.*]] = fir.embox %[[VAL_12]]#0 : (!fir.ref<!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>) -> !fir.box<!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>
! CHECK:           %[[VAL_14:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]*}}>>
! CHECK:           %[[VAL_15:.*]] = arith.constant {{[0-9]*}} : i32
! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_13]] : (!fir.box<!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>) -> !fir.box<none>
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_14]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_18:.*]] = fir.call @_FortranAInitialize(%[[VAL_16]], %[[VAL_17]], %[[VAL_15]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:           %[[VAL_19:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_20:.*]] = hlfir.designate %[[VAL_12]]#0{"c"}   typeparams %[[VAL_19]] {fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>, index) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>>
! CHECK:           %[[VAL_21:.*]] = fir.load %[[VAL_11]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>>
! CHECK:           %[[VAL_22:.*]] = arith.constant 2 : i64
! CHECK:           %[[VAL_23:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_24:.*]]:3 = fir.box_dims %[[VAL_21]], %[[VAL_23]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_25:.*]] = fir.shape %[[VAL_24]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_26:.*]] = hlfir.elemental %[[VAL_25]] typeparams %[[VAL_22]] : (!fir.shape<1>, i64) -> !hlfir.expr<?x!fir.char<1,?>> {
! CHECK:           ^bb0(%[[VAL_27:.*]]: index):
! CHECK:             %[[VAL_28:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_29:.*]]:3 = fir.box_dims %[[VAL_21]], %[[VAL_28]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>, index) -> (index, index, index)
! CHECK:             %[[VAL_30:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_31:.*]] = arith.subi %[[VAL_29]]#0, %[[VAL_30]] : index
! CHECK:             %[[VAL_32:.*]] = arith.addi %[[VAL_27]], %[[VAL_31]] : index
! CHECK:             %[[VAL_33:.*]] = hlfir.designate %[[VAL_21]] (%[[VAL_32]])  typeparams %[[VAL_10]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>, index, index) -> !fir.ref<!fir.char<1,2>>
! CHECK:             %[[VAL_34:.*]] = hlfir.set_length %[[VAL_33]] len %[[VAL_22]] : (!fir.ref<!fir.char<1,2>>, i64) -> !hlfir.expr<!fir.char<1,2>>
! CHECK:             hlfir.yield_element %[[VAL_34]] : !hlfir.expr<!fir.char<1,2>>
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_35:.*]] to %[[VAL_20]] realloc : !hlfir.expr<?x!fir.char<1,?>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>>
! CHECK:           hlfir.assign %[[VAL_12]]#0 to %[[VAL_3]]#0 : !fir.ref<!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>, !fir.ref<!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>
! CHECK:           hlfir.destroy %[[VAL_35]] : !hlfir.expr<?x!fir.char<1,?>>
! CHECK:           return
! CHECK:         }

subroutine test5(x)
  use types
  type(t4), allocatable :: x(:)
  type(t5) res
  res = t5(x)
end subroutine test5
! CHECK-LABEL:   func.func @_QPtest5(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}> {bindc_name = "res", uniq_name = "_QFtest5Eres"}
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFtest5Eres"} : (!fir.ref<!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>>) -> (!fir.ref<!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>>, !fir.ref<!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>>)
! CHECK:           %[[VAL_4:.*]] = fir.embox %[[VAL_3]]#1 : (!fir.ref<!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>>) -> !fir.box<!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>>
! CHECK:           %[[VAL_5:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]*}}>>
! CHECK:           %[[VAL_6:.*]] = arith.constant {{[0-9]*}} : i32
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_4]] : (!fir.box<!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>>) -> !fir.box<none>
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_5]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_9:.*]] = fir.call @_FortranAInitialize(%[[VAL_7]], %[[VAL_8]], %[[VAL_6]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest5Ex"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>>)
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "ctor.temp"} : (!fir.ref<!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>>) -> (!fir.ref<!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>>, !fir.ref<!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>>)
! CHECK:           %[[VAL_12:.*]] = fir.embox %[[VAL_11]]#0 : (!fir.ref<!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>>) -> !fir.box<!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>>
! CHECK:           %[[VAL_13:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]*}}>>
! CHECK:           %[[VAL_14:.*]] = arith.constant {{[0-9]*}} : i32
! CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_12]] : (!fir.box<!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>>) -> !fir.box<none>
! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_13]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_17:.*]] = fir.call @_FortranAInitialize(%[[VAL_15]], %[[VAL_16]], %[[VAL_14]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:           %[[VAL_18:.*]] = hlfir.designate %[[VAL_11]]#0{"t5m"}   {fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>>
! CHECK:           hlfir.assign %[[VAL_10]]#0 to %[[VAL_18]] realloc : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>>
! CHECK:           hlfir.assign %[[VAL_11]]#0 to %[[VAL_3]]#0 : !fir.ref<!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>>, !fir.ref<!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>>
! CHECK:           return
! CHECK:         }

subroutine test6(x, c)
  use types
  type(t4), allocatable :: x(:)
  character(4) :: c
  type(t6) res
  res = t6(t5(x), [t1(c)])
end subroutine test6
! CHECK-LABEL:   func.func @_QPtest6(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>> {fir.bindc_name = "x"},
! CHECK-SAME:                        %[[VAL_1:.*]]: !fir.boxchar<1> {fir.bindc_name = "c"}) {
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.type<_QMtypesTt1{c:!fir.char<1,4>}>
! CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.array<10xi64> {bindc_name = ".rt.arrayctor.vector"}
! CHECK:           %[[VAL_4:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>>> {bindc_name = ".tmp.arrayctor"}
! CHECK:           %[[VAL_5:.*]] = fir.alloca !fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>
! CHECK:           %[[VAL_6:.*]] = fir.alloca !fir.type<_QMtypesTt6{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>,t6m:!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>}>
! CHECK:           %[[VAL_7:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_8:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_7]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,4>>
! CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_9]] typeparams %[[VAL_8]] {uniq_name = "_QFtest6Ec"} : (!fir.ref<!fir.char<1,4>>, index) -> (!fir.ref<!fir.char<1,4>>, !fir.ref<!fir.char<1,4>>)
! CHECK:           %[[VAL_11:.*]] = fir.alloca !fir.type<_QMtypesTt6{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>,t6m:!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>}> {bindc_name = "res", uniq_name = "_QFtest6Eres"}
! CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_11]] {uniq_name = "_QFtest6Eres"} : (!fir.ref<!fir.type<_QMtypesTt6{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>,t6m:!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>}>>) -> (!fir.ref<!fir.type<_QMtypesTt6{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>,t6m:!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>}>>, !fir.ref<!fir.type<_QMtypesTt6{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>,t6m:!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>}>>)
! CHECK:           %[[VAL_13:.*]] = fir.embox %[[VAL_12]]#1 : (!fir.ref<!fir.type<_QMtypesTt6{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>,t6m:!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>}>>) -> !fir.box<!fir.type<_QMtypesTt6{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>,t6m:!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>}>>
! CHECK:           %[[VAL_14:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]*}}>>
! CHECK:           %[[VAL_15:.*]] = arith.constant {{[0-9]*}} : i32
! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_13]] : (!fir.box<!fir.type<_QMtypesTt6{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>,t6m:!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>}>>) -> !fir.box<none>
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_14]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_18:.*]] = fir.call @_FortranAInitialize(%[[VAL_16]], %[[VAL_17]], %[[VAL_15]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:           %[[VAL_19:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest6Ex"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>>)
! CHECK:           %[[VAL_20:.*]]:2 = hlfir.declare %[[VAL_6]] {uniq_name = "ctor.temp"} : (!fir.ref<!fir.type<_QMtypesTt6{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>,t6m:!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>}>>) -> (!fir.ref<!fir.type<_QMtypesTt6{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>,t6m:!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>}>>, !fir.ref<!fir.type<_QMtypesTt6{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>,t6m:!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>}>>)
! CHECK:           %[[VAL_21:.*]] = fir.embox %[[VAL_20]]#0 : (!fir.ref<!fir.type<_QMtypesTt6{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>,t6m:!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>}>>) -> !fir.box<!fir.type<_QMtypesTt6{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>,t6m:!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>}>>
! CHECK:           %[[VAL_22:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]*}}>>
! CHECK:           %[[VAL_23:.*]] = arith.constant {{[0-9]*}} : i32
! CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_21]] : (!fir.box<!fir.type<_QMtypesTt6{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>,t6m:!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>}>>) -> !fir.box<none>
! CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_22]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_26:.*]] = fir.call @_FortranAInitialize(%[[VAL_24]], %[[VAL_25]], %[[VAL_23]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:           %[[VAL_27:.*]] = hlfir.parent_comp %[[VAL_20]]#0 : (!fir.ref<!fir.type<_QMtypesTt6{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>,t6m:!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>}>>) -> !fir.ref<!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>>
! CHECK:           %[[VAL_28:.*]]:2 = hlfir.declare %[[VAL_5]] {uniq_name = "ctor.temp"} : (!fir.ref<!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>>) -> (!fir.ref<!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>>, !fir.ref<!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>>)
! CHECK:           %[[VAL_29:.*]] = fir.embox %[[VAL_28]]#0 : (!fir.ref<!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>>) -> !fir.box<!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>>
! CHECK:           %[[VAL_30:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]*}}>>
! CHECK:           %[[VAL_31:.*]] = arith.constant {{[0-9]*}} : i32
! CHECK:           %[[VAL_32:.*]] = fir.convert %[[VAL_29]] : (!fir.box<!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>>) -> !fir.box<none>
! CHECK:           %[[VAL_33:.*]] = fir.convert %[[VAL_30]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_34:.*]] = fir.call @_FortranAInitialize(%[[VAL_32]], %[[VAL_33]], %[[VAL_31]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:           %[[VAL_35:.*]] = hlfir.designate %[[VAL_28]]#0{"t5m"}   {fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>>
! CHECK:           hlfir.assign %[[VAL_19]]#0 to %[[VAL_35]] realloc : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>>
! CHECK:           hlfir.assign %[[VAL_28]]#0 to %[[VAL_27]] : !fir.ref<!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>>, !fir.ref<!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>>
! CHECK:           %[[VAL_36:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_37:.*]] = fir.shape %[[VAL_36]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_38:.*]] = hlfir.designate %[[VAL_20]]#0{"t6m"} <%[[VAL_37]]>   shape %[[VAL_37]] : (!fir.ref<!fir.type<_QMtypesTt6{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>,t6m:!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>}>>, !fir.shape<1>, !fir.shape<1>) -> !fir.ref<!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>>
! CHECK:           %[[VAL_39:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_40:.*]] = fir.allocmem !fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>> {bindc_name = ".tmp.arrayctor", uniq_name = ""}
! CHECK:           %[[VAL_41:.*]] = fir.shape %[[VAL_39]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_42:.*]]:2 = hlfir.declare %[[VAL_40]](%[[VAL_41]]) {uniq_name = ".tmp.arrayctor"} : (!fir.heap<!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>>, !fir.shape<1>) -> (!fir.heap<!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>>, !fir.heap<!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>>)
! CHECK:           %[[VAL_43:.*]] = fir.embox %[[VAL_42]]#1(%[[VAL_41]]) : (!fir.heap<!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>>>
! CHECK:           fir.store %[[VAL_43]] to %[[VAL_4]] : !fir.ref<!fir.box<!fir.heap<!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>>>>
! CHECK:           %[[VAL_44:.*]] = arith.constant false
! CHECK:           %[[VAL_45:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.array<10xi64>>) -> !fir.llvm_ptr<i8>
! CHECK:           %[[VAL_46:.*]] = arith.constant 80 : i32
! CHECK:           %[[VAL_47:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]*}}>>
! CHECK:           %[[VAL_48:.*]] = arith.constant {{[0-9]*}} : i32
! CHECK:           %[[VAL_49:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_50:.*]] = fir.convert %[[VAL_47]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_51:.*]] = fir.call @_FortranAInitArrayConstructorVector(%[[VAL_45]], %[[VAL_49]], %[[VAL_44]], %[[VAL_46]], %[[VAL_50]], %[[VAL_48]]) fastmath<contract> : (!fir.llvm_ptr<i8>, !fir.ref<!fir.box<none>>, i1, i32, !fir.ref<i8>, i32) -> none
! CHECK:           %[[VAL_52:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "ctor.temp"} : (!fir.ref<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>) -> (!fir.ref<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>, !fir.ref<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>)
! CHECK:           %[[VAL_53:.*]] = fir.embox %[[VAL_52]]#0 : (!fir.ref<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>) -> !fir.box<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>
! CHECK:           %[[VAL_54:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]*}}>>
! CHECK:           %[[VAL_55:.*]] = arith.constant {{[0-9]*}} : i32
! CHECK:           %[[VAL_56:.*]] = fir.convert %[[VAL_53]] : (!fir.box<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>) -> !fir.box<none>
! CHECK:           %[[VAL_57:.*]] = fir.convert %[[VAL_54]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_58:.*]] = fir.call @_FortranAInitialize(%[[VAL_56]], %[[VAL_57]], %[[VAL_55]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:           %[[VAL_59:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_60:.*]] = hlfir.designate %[[VAL_52]]#0{"c"}   typeparams %[[VAL_59]] : (!fir.ref<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>, index) -> !fir.ref<!fir.char<1,4>>
! CHECK:           %[[VAL_61:.*]] = arith.constant 4 : i64
! CHECK:           %[[VAL_62:.*]] = hlfir.set_length %[[VAL_10]]#0 len %[[VAL_61]] : (!fir.ref<!fir.char<1,4>>, i64) -> !hlfir.expr<!fir.char<1,4>>
! CHECK:           hlfir.assign %[[VAL_62]] to %[[VAL_60]] : !hlfir.expr<!fir.char<1,4>>, !fir.ref<!fir.char<1,4>>
! CHECK:           %[[VAL_63:.*]] = fir.convert %[[VAL_52]]#1 : (!fir.ref<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>) -> !fir.llvm_ptr<i8>
! CHECK:           %[[VAL_64:.*]] = fir.call @_FortranAPushArrayConstructorSimpleScalar(%[[VAL_45]], %[[VAL_63]]) fastmath<contract> : (!fir.llvm_ptr<i8>, !fir.llvm_ptr<i8>) -> none
! CHECK:           %[[VAL_65:.*]] = arith.constant true
! CHECK:           %[[VAL_66:.*]] = hlfir.as_expr %[[VAL_42]]#0 move %[[VAL_65]] : (!fir.heap<!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>>, i1) -> !hlfir.expr<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>
! CHECK:           hlfir.assign %[[VAL_66]] to %[[VAL_38]] : !hlfir.expr<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>, !fir.ref<!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>>
! CHECK:           hlfir.assign %[[VAL_20]]#0 to %[[VAL_12]]#0 : !fir.ref<!fir.type<_QMtypesTt6{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>,t6m:!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>}>>, !fir.ref<!fir.type<_QMtypesTt6{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>,t6m:!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>}>>
! CHECK:           hlfir.destroy %[[VAL_66]] : !hlfir.expr<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>
! CHECK:           return
! CHECK:         }

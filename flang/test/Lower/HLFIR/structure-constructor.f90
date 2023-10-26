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
  type t7
     integer :: c1
     real, allocatable :: c2(:)
  end type t7
  type t8
     character(11), allocatable :: c
  end type t8
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
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,4>>
! CHECK:           %[[VAL_6:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_5]] typeparams %[[VAL_6]] {uniq_name = "_QFtest1Ex"} : (!fir.ref<!fir.char<1,4>>, index) -> (!fir.ref<!fir.char<1,4>>, !fir.ref<!fir.char<1,4>>)
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "ctor.temp"} : (!fir.ref<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>) -> (!fir.ref<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>, !fir.ref<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>)
! CHECK:           %[[VAL_9:.*]] = fir.embox %[[VAL_8]]#0 : (!fir.ref<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>) -> !fir.box<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>
! CHECK:           %[[VAL_10:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]*}}>>
! CHECK:           %[[VAL_11:.*]] = arith.constant {{[0-9]*}} : i32
! CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_9]] : (!fir.box<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>) -> !fir.box<none>
! CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_10]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_14:.*]] = fir.call @_FortranAInitialize(%[[VAL_12]], %[[VAL_13]], %[[VAL_11]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:           %[[VAL_15:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_16:.*]] = hlfir.designate %[[VAL_8]]#0{"c"}   typeparams %[[VAL_15]] : (!fir.ref<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>, index) -> !fir.ref<!fir.char<1,4>>
! CHECK:           hlfir.assign %[[VAL_7]]#0 to %[[VAL_16]] temporary_lhs : !fir.ref<!fir.char<1,4>>, !fir.ref<!fir.char<1,4>>
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
! CHECK:           hlfir.assign %[[VAL_6]]#0 to %[[VAL_16]] temporary_lhs : !fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>
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
! CHECK:           %[[VAL_21:.*]] = fir.load %[[VAL_11]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>>
! CHECK:           %[[VAL_22:.*]] = fir.box_addr %[[VAL_21]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>) -> !fir.heap<!fir.array<?x!fir.char<1,2>>>
! CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (!fir.heap<!fir.array<?x!fir.char<1,2>>>) -> i64
! CHECK:           %[[VAL_24:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_25:.*]] = arith.cmpi ne, %[[VAL_23]], %[[VAL_24]] : i64
! CHECK:           fir.if %[[VAL_25]] {
! CHECK:             %[[VAL_26:.*]] = fir.load %[[VAL_11]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>>
! CHECK:             hlfir.assign %[[VAL_26]] to %[[VAL_20]] realloc keep_lhs_len temporary_lhs : !fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>>
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_12]]#0 to %[[VAL_3]]#0 : !fir.ref<!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>, !fir.ref<!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>
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
! CHECK:           %[[VAL_19:.*]] = fir.load %[[VAL_10]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>>
! CHECK:           %[[VAL_20:.*]] = fir.box_addr %[[VAL_19]] : (!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>) -> !fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>
! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>) -> i64
! CHECK:           %[[VAL_22:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_23:.*]] = arith.cmpi ne, %[[VAL_21]], %[[VAL_22]] : i64
! CHECK:           fir.if %[[VAL_23]] {
! CHECK:             %[[VAL_24:.*]] = fir.load %[[VAL_10]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>>
! CHECK:             hlfir.assign %[[VAL_24]] to %[[VAL_18]] realloc temporary_lhs : !fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>>
! CHECK:           }
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
! CHECK:           %[[VAL_6:.*]] = fir.alloca !fir.type<_QMtypesTt6{t5:!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>,t6m:!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>}>
! CHECK:           %[[VAL_7:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,4>>
! CHECK:           %[[VAL_9:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_8]] typeparams %[[VAL_9]] {uniq_name = "_QFtest6Ec"} : (!fir.ref<!fir.char<1,4>>, index) -> (!fir.ref<!fir.char<1,4>>, !fir.ref<!fir.char<1,4>>)
! CHECK:           %[[VAL_11:.*]] = fir.alloca !fir.type<_QMtypesTt6{t5:!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>,t6m:!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>}> {bindc_name = "res", uniq_name = "_QFtest6Eres"}
! CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_11]] {uniq_name = "_QFtest6Eres"} : (!fir.ref<!fir.type<_QMtypesTt6{t5:!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>,t6m:!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>}>>) -> (!fir.ref<!fir.type<_QMtypesTt6{t5:!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>,t6m:!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>}>>, !fir.ref<!fir.type<_QMtypesTt6{t5:!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>,t6m:!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>}>>)
! CHECK:           %[[VAL_13:.*]] = fir.embox %[[VAL_12]]#1 : (!fir.ref<!fir.type<_QMtypesTt6{t5:!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>,t6m:!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>}>>) -> !fir.box<!fir.type<_QMtypesTt6{t5:!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>,t6m:!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>}>>
! CHECK:           %[[VAL_14:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]*}}>>
! CHECK:           %[[VAL_15:.*]] = arith.constant {{[0-9]*}} : i32
! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_13]] : (!fir.box<!fir.type<_QMtypesTt6{t5:!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>,t6m:!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>}>>) -> !fir.box<none>
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_14]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_18:.*]] = fir.call @_FortranAInitialize(%[[VAL_16]], %[[VAL_17]], %[[VAL_15]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:           %[[VAL_19:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest6Ex"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>>)
! CHECK:           %[[VAL_20:.*]]:2 = hlfir.declare %[[VAL_6]] {uniq_name = "ctor.temp"} : (!fir.ref<!fir.type<_QMtypesTt6{t5:!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>,t6m:!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>}>>) -> (!fir.ref<!fir.type<_QMtypesTt6{t5:!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>,t6m:!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>}>>, !fir.ref<!fir.type<_QMtypesTt6{t5:!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>,t6m:!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>}>>)
! CHECK:           %[[VAL_21:.*]] = fir.embox %[[VAL_20]]#0 : (!fir.ref<!fir.type<_QMtypesTt6{t5:!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>,t6m:!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>}>>) -> !fir.box<!fir.type<_QMtypesTt6{t5:!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>,t6m:!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>}>>
! CHECK:           %[[VAL_22:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]*}}>>
! CHECK:           %[[VAL_23:.*]] = arith.constant {{[0-9]*}} : i32
! CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_21]] : (!fir.box<!fir.type<_QMtypesTt6{t5:!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>,t6m:!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>}>>) -> !fir.box<none>
! CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_22]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_26:.*]] = fir.call @_FortranAInitialize(%[[VAL_24]], %[[VAL_25]], %[[VAL_23]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:           %[[VAL_27:.*]] = hlfir.designate %[[VAL_20]]#0{"t5"}   : (!fir.ref<!fir.type<_QMtypesTt6{t5:!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>,t6m:!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>}>>) -> !fir.ref<!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>>
! CHECK:           %[[VAL_28:.*]]:2 = hlfir.declare %[[VAL_5]] {uniq_name = "ctor.temp"} : (!fir.ref<!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>>) -> (!fir.ref<!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>>, !fir.ref<!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>>)
! CHECK:           %[[VAL_29:.*]] = fir.embox %[[VAL_28]]#0 : (!fir.ref<!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>>) -> !fir.box<!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>>
! CHECK:           %[[VAL_30:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]*}}>>
! CHECK:           %[[VAL_31:.*]] = arith.constant {{[0-9]*}} : i32
! CHECK:           %[[VAL_32:.*]] = fir.convert %[[VAL_29]] : (!fir.box<!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>>) -> !fir.box<none>
! CHECK:           %[[VAL_33:.*]] = fir.convert %[[VAL_30]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_34:.*]] = fir.call @_FortranAInitialize(%[[VAL_32]], %[[VAL_33]], %[[VAL_31]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:           %[[VAL_35:.*]] = hlfir.designate %[[VAL_28]]#0{"t5m"}   {fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>>
! CHECK:           %[[VAL_36:.*]] = fir.load %[[VAL_19]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>>
! CHECK:           %[[VAL_37:.*]] = fir.box_addr %[[VAL_36]] : (!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>) -> !fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>
! CHECK:           %[[VAL_38:.*]] = fir.convert %[[VAL_37]] : (!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>) -> i64
! CHECK:           %[[VAL_39:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_40:.*]] = arith.cmpi ne, %[[VAL_38]], %[[VAL_39]] : i64
! CHECK:           fir.if %[[VAL_40]] {
! CHECK:             %[[VAL_41:.*]] = fir.load %[[VAL_19]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>>
! CHECK:             hlfir.assign %[[VAL_41]] to %[[VAL_35]] realloc temporary_lhs : !fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>>
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_28]]#0 to %[[VAL_27]] temporary_lhs : !fir.ref<!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>>, !fir.ref<!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>>
! CHECK:           %[[VAL_42:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_43:.*]] = fir.shape %[[VAL_42]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_44:.*]] = hlfir.designate %[[VAL_20]]#0{"t6m"} <%[[VAL_43]]>   shape %[[VAL_43]] : (!fir.ref<!fir.type<_QMtypesTt6{t5:!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>,t6m:!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>}>>, !fir.shape<1>, !fir.shape<1>) -> !fir.ref<!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>>
! CHECK:           %[[VAL_45:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_46:.*]] = fir.allocmem !fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>> {bindc_name = ".tmp.arrayctor", uniq_name = ""}
! CHECK:           %[[VAL_47:.*]] = fir.shape %[[VAL_45]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_48:.*]]:2 = hlfir.declare %[[VAL_46]](%[[VAL_47]]) {uniq_name = ".tmp.arrayctor"} : (!fir.heap<!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>>, !fir.shape<1>) -> (!fir.heap<!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>>, !fir.heap<!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>>)
! CHECK:           %[[VAL_49:.*]] = fir.embox %[[VAL_48]]#1(%[[VAL_47]]) : (!fir.heap<!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>>>
! CHECK:           fir.store %[[VAL_49]] to %[[VAL_4]] : !fir.ref<!fir.box<!fir.heap<!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>>>>
! CHECK:           %[[VAL_50:.*]] = arith.constant false
! CHECK:           %[[VAL_51:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.array<10xi64>>) -> !fir.llvm_ptr<i8>
! CHECK:           %[[VAL_52:.*]] = arith.constant 80 : i32
! CHECK:           %[[VAL_53:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]*}}>>
! CHECK:           %[[VAL_54:.*]] = arith.constant {{[0-9]*}} : i32
! CHECK:           %[[VAL_55:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_56:.*]] = fir.convert %[[VAL_53]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_57:.*]] = fir.call @_FortranAInitArrayConstructorVector(%[[VAL_51]], %[[VAL_55]], %[[VAL_50]], %[[VAL_52]], %[[VAL_56]], %[[VAL_54]]) fastmath<contract> : (!fir.llvm_ptr<i8>, !fir.ref<!fir.box<none>>, i1, i32, !fir.ref<i8>, i32) -> none
! CHECK:           %[[VAL_58:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "ctor.temp"} : (!fir.ref<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>) -> (!fir.ref<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>, !fir.ref<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>)
! CHECK:           %[[VAL_59:.*]] = fir.embox %[[VAL_58]]#0 : (!fir.ref<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>) -> !fir.box<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>
! CHECK:           %[[VAL_60:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]*}}>>
! CHECK:           %[[VAL_61:.*]] = arith.constant {{[0-9]*}} : i32
! CHECK:           %[[VAL_62:.*]] = fir.convert %[[VAL_59]] : (!fir.box<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>) -> !fir.box<none>
! CHECK:           %[[VAL_63:.*]] = fir.convert %[[VAL_60]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_64:.*]] = fir.call @_FortranAInitialize(%[[VAL_62]], %[[VAL_63]], %[[VAL_61]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:           %[[VAL_65:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_66:.*]] = hlfir.designate %[[VAL_58]]#0{"c"}   typeparams %[[VAL_65]] : (!fir.ref<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>, index) -> !fir.ref<!fir.char<1,4>>
! CHECK:           hlfir.assign %[[VAL_10]]#0 to %[[VAL_66]] temporary_lhs : !fir.ref<!fir.char<1,4>>, !fir.ref<!fir.char<1,4>>
! CHECK:           %[[VAL_67:.*]] = fir.convert %[[VAL_58]]#1 : (!fir.ref<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>) -> !fir.llvm_ptr<i8>
! CHECK:           %[[VAL_68:.*]] = fir.call @_FortranAPushArrayConstructorSimpleScalar(%[[VAL_51]], %[[VAL_67]]) fastmath<contract> : (!fir.llvm_ptr<i8>, !fir.llvm_ptr<i8>) -> none
! CHECK:           %[[VAL_69:.*]] = arith.constant true
! CHECK:           %[[VAL_70:.*]] = hlfir.as_expr %[[VAL_48]]#0 move %[[VAL_69]] : (!fir.heap<!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>>, i1) -> !hlfir.expr<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>
! CHECK:           hlfir.assign %[[VAL_70]] to %[[VAL_44]] temporary_lhs : !hlfir.expr<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>, !fir.ref<!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>>
! CHECK:           hlfir.assign %[[VAL_20]]#0 to %[[VAL_12]]#0 : !fir.ref<!fir.type<_QMtypesTt6{t5:!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>,t6m:!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>}>>, !fir.ref<!fir.type<_QMtypesTt6{t5:!fir.type<_QMtypesTt5{t5m:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt4{c:!fir.box<!fir.heap<!fir.array<?x!fir.char<1,2>>>>}>>>>}>,t6m:!fir.array<1x!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>}>>
! CHECK:           return
! CHECK:         }

! Test that NULL() expression passed as the component "value"
! for the missing component initializer in the structure constructor
! is handled properly: the component is just left unallocated with its
! defined rank and there is no hlfir.assign for this part
! of the constructor.
subroutine test7(n)
  use types
  integer, intent(in) :: n
  type(t7) :: x
  x = t7(n)
end subroutine test7
! CHECK-LABEL:   func.func @_QPtest7(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.type<_QMtypesTt7{c1:i32,c2:!fir.box<!fir.heap<!fir.array<?xf32>>>}>
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtest7En"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.type<_QMtypesTt7{c1:i32,c2:!fir.box<!fir.heap<!fir.array<?xf32>>>}> {bindc_name = "x", uniq_name = "_QFtest7Ex"}
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] {uniq_name = "_QFtest7Ex"} : (!fir.ref<!fir.type<_QMtypesTt7{c1:i32,c2:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> (!fir.ref<!fir.type<_QMtypesTt7{c1:i32,c2:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>, !fir.ref<!fir.type<_QMtypesTt7{c1:i32,c2:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>)
! CHECK:           %[[VAL_5:.*]] = fir.embox %[[VAL_4]]#1 : (!fir.ref<!fir.type<_QMtypesTt7{c1:i32,c2:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> !fir.box<!fir.type<_QMtypesTt7{c1:i32,c2:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
! CHECK:           %[[VAL_6:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]*}}>>
! CHECK:           %[[VAL_7:.*]] = arith.constant {{[0-9]*}} : i32
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_5]] : (!fir.box<!fir.type<_QMtypesTt7{c1:i32,c2:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> !fir.box<none>
! CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_10:.*]] = fir.call @_FortranAInitialize(%[[VAL_8]], %[[VAL_9]], %[[VAL_7]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "ctor.temp"} : (!fir.ref<!fir.type<_QMtypesTt7{c1:i32,c2:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> (!fir.ref<!fir.type<_QMtypesTt7{c1:i32,c2:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>, !fir.ref<!fir.type<_QMtypesTt7{c1:i32,c2:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>)
! CHECK:           %[[VAL_12:.*]] = fir.embox %[[VAL_11]]#0 : (!fir.ref<!fir.type<_QMtypesTt7{c1:i32,c2:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> !fir.box<!fir.type<_QMtypesTt7{c1:i32,c2:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
! CHECK:           %[[VAL_13:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]*}}>>
! CHECK:           %[[VAL_14:.*]] = arith.constant {{[0-9]*}} : i32
! CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_12]] : (!fir.box<!fir.type<_QMtypesTt7{c1:i32,c2:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> !fir.box<none>
! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_13]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_17:.*]] = fir.call @_FortranAInitialize(%[[VAL_15]], %[[VAL_16]], %[[VAL_14]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:           %[[VAL_18:.*]] = hlfir.designate %[[VAL_11]]#0{"c1"}   : (!fir.ref<!fir.type<_QMtypesTt7{c1:i32,c2:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> !fir.ref<i32>
! CHECK:           %[[VAL_19:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<i32>
! CHECK:           hlfir.assign %[[VAL_19]] to %[[VAL_18]] temporary_lhs : i32, !fir.ref<i32>
! CHECK:           hlfir.assign %[[VAL_11]]#0 to %[[VAL_4]]#0 : !fir.ref<!fir.type<_QMtypesTt7{c1:i32,c2:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>, !fir.ref<!fir.type<_QMtypesTt7{c1:i32,c2:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
! CHECK:           return
! CHECK:         }

! Test character allocatable component initialization
! from character allocatable of different size.
subroutine test8
  use types
  character(12), allocatable :: x
  type(t8) res
  res = t8(x)
end subroutine test8
! CHECK-LABEL:   func.func @_QPtest8() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.type<_QMtypesTt8{c:!fir.box<!fir.heap<!fir.char<1,11>>>}>
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.type<_QMtypesTt8{c:!fir.box<!fir.heap<!fir.char<1,11>>>}> {bindc_name = "res", uniq_name = "_QFtest8Eres"}
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFtest8Eres"} : (!fir.ref<!fir.type<_QMtypesTt8{c:!fir.box<!fir.heap<!fir.char<1,11>>>}>>) -> (!fir.ref<!fir.type<_QMtypesTt8{c:!fir.box<!fir.heap<!fir.char<1,11>>>}>>, !fir.ref<!fir.type<_QMtypesTt8{c:!fir.box<!fir.heap<!fir.char<1,11>>>}>>)
! CHECK:           %[[VAL_3:.*]] = fir.embox %[[VAL_2]]#1 : (!fir.ref<!fir.type<_QMtypesTt8{c:!fir.box<!fir.heap<!fir.char<1,11>>>}>>) -> !fir.box<!fir.type<_QMtypesTt8{c:!fir.box<!fir.heap<!fir.char<1,11>>>}>>
! CHECK:           %[[VAL_4:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]*}}>>
! CHECK:           %[[VAL_5:.*]] = arith.constant {{[0-9]*}} : i32
! CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_3]] : (!fir.box<!fir.type<_QMtypesTt8{c:!fir.box<!fir.heap<!fir.char<1,11>>>}>>) -> !fir.box<none>
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_8:.*]] = fir.call @_FortranAInitialize(%[[VAL_6]], %[[VAL_7]], %[[VAL_5]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:           %[[VAL_9:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,12>>> {bindc_name = "x", uniq_name = "_QFtest8Ex"}
! CHECK:           %[[VAL_10:.*]] = arith.constant 12 : index
! CHECK:           %[[VAL_11:.*]] = fir.zero_bits !fir.heap<!fir.char<1,12>>
! CHECK:           %[[VAL_12:.*]] = fir.embox %[[VAL_11]] : (!fir.heap<!fir.char<1,12>>) -> !fir.box<!fir.heap<!fir.char<1,12>>>
! CHECK:           fir.store %[[VAL_12]] to %[[VAL_9]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,12>>>>
! CHECK:           %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_9]] typeparams %[[VAL_10]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest8Ex"} : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,12>>>>, index) -> (!fir.ref<!fir.box<!fir.heap<!fir.char<1,12>>>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,12>>>>)
! CHECK:           %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "ctor.temp"} : (!fir.ref<!fir.type<_QMtypesTt8{c:!fir.box<!fir.heap<!fir.char<1,11>>>}>>) -> (!fir.ref<!fir.type<_QMtypesTt8{c:!fir.box<!fir.heap<!fir.char<1,11>>>}>>, !fir.ref<!fir.type<_QMtypesTt8{c:!fir.box<!fir.heap<!fir.char<1,11>>>}>>)
! CHECK:           %[[VAL_15:.*]] = fir.embox %[[VAL_14]]#0 : (!fir.ref<!fir.type<_QMtypesTt8{c:!fir.box<!fir.heap<!fir.char<1,11>>>}>>) -> !fir.box<!fir.type<_QMtypesTt8{c:!fir.box<!fir.heap<!fir.char<1,11>>>}>>
! CHECK:           %[[VAL_16:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]*}}>>
! CHECK:           %[[VAL_17:.*]] = arith.constant {{[0-9]*}} : i32
! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_15]] : (!fir.box<!fir.type<_QMtypesTt8{c:!fir.box<!fir.heap<!fir.char<1,11>>>}>>) -> !fir.box<none>
! CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_16]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_20:.*]] = fir.call @_FortranAInitialize(%[[VAL_18]], %[[VAL_19]], %[[VAL_17]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:           %[[VAL_21:.*]] = arith.constant 11 : index
! CHECK:           %[[VAL_22:.*]] = hlfir.designate %[[VAL_14]]#0{"c"}   typeparams %[[VAL_21]] {fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QMtypesTt8{c:!fir.box<!fir.heap<!fir.char<1,11>>>}>>, index) -> !fir.ref<!fir.box<!fir.heap<!fir.char<1,11>>>>
! CHECK:           %[[VAL_23:.*]] = fir.load %[[VAL_13]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,12>>>>
! CHECK:           %[[VAL_24:.*]] = fir.box_addr %[[VAL_23]] : (!fir.box<!fir.heap<!fir.char<1,12>>>) -> !fir.heap<!fir.char<1,12>>
! CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (!fir.heap<!fir.char<1,12>>) -> i64
! CHECK:           %[[VAL_26:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_27:.*]] = arith.cmpi ne, %[[VAL_25]], %[[VAL_26]] : i64
! CHECK:           fir.if %[[VAL_27]] {
! CHECK:             %[[VAL_28:.*]] = fir.load %[[VAL_13]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,12>>>>
! CHECK:             %[[VAL_29:.*]] = fir.box_addr %[[VAL_28]] : (!fir.box<!fir.heap<!fir.char<1,12>>>) -> !fir.heap<!fir.char<1,12>>
! CHECK:             hlfir.assign %[[VAL_29]] to %[[VAL_22]] realloc keep_lhs_len temporary_lhs : !fir.heap<!fir.char<1,12>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,11>>>>
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_14]]#0 to %[[VAL_2]]#0 : !fir.ref<!fir.type<_QMtypesTt8{c:!fir.box<!fir.heap<!fir.char<1,11>>>}>>, !fir.ref<!fir.type<_QMtypesTt8{c:!fir.box<!fir.heap<!fir.char<1,11>>>}>>
! CHECK:           return
! CHECK:         }

! Test character allocatable component initialization
! from character non-allocatable of different size.
subroutine test9
  use types
  character(12) :: x
  type(t8) res
  res = t8(x)
end subroutine test9
! CHECK-LABEL:   func.func @_QPtest9() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.type<_QMtypesTt8{c:!fir.box<!fir.heap<!fir.char<1,11>>>}>
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.type<_QMtypesTt8{c:!fir.box<!fir.heap<!fir.char<1,11>>>}> {bindc_name = "res", uniq_name = "_QFtest9Eres"}
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFtest9Eres"} : (!fir.ref<!fir.type<_QMtypesTt8{c:!fir.box<!fir.heap<!fir.char<1,11>>>}>>) -> (!fir.ref<!fir.type<_QMtypesTt8{c:!fir.box<!fir.heap<!fir.char<1,11>>>}>>, !fir.ref<!fir.type<_QMtypesTt8{c:!fir.box<!fir.heap<!fir.char<1,11>>>}>>)
! CHECK:           %[[VAL_3:.*]] = fir.embox %[[VAL_2]]#1 : (!fir.ref<!fir.type<_QMtypesTt8{c:!fir.box<!fir.heap<!fir.char<1,11>>>}>>) -> !fir.box<!fir.type<_QMtypesTt8{c:!fir.box<!fir.heap<!fir.char<1,11>>>}>>
! CHECK:           %[[VAL_4:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]*}}>>
! CHECK:           %[[VAL_5:.*]] = arith.constant {{[0-9]*}} : i32
! CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_3]] : (!fir.box<!fir.type<_QMtypesTt8{c:!fir.box<!fir.heap<!fir.char<1,11>>>}>>) -> !fir.box<none>
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_8:.*]] = fir.call @_FortranAInitialize(%[[VAL_6]], %[[VAL_7]], %[[VAL_5]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:           %[[VAL_9:.*]] = arith.constant 12 : index
! CHECK:           %[[VAL_10:.*]] = fir.alloca !fir.char<1,12> {bindc_name = "x", uniq_name = "_QFtest9Ex"}
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_10]] typeparams %[[VAL_9]] {uniq_name = "_QFtest9Ex"} : (!fir.ref<!fir.char<1,12>>, index) -> (!fir.ref<!fir.char<1,12>>, !fir.ref<!fir.char<1,12>>)
! CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "ctor.temp"} : (!fir.ref<!fir.type<_QMtypesTt8{c:!fir.box<!fir.heap<!fir.char<1,11>>>}>>) -> (!fir.ref<!fir.type<_QMtypesTt8{c:!fir.box<!fir.heap<!fir.char<1,11>>>}>>, !fir.ref<!fir.type<_QMtypesTt8{c:!fir.box<!fir.heap<!fir.char<1,11>>>}>>)
! CHECK:           %[[VAL_13:.*]] = fir.embox %[[VAL_12]]#0 : (!fir.ref<!fir.type<_QMtypesTt8{c:!fir.box<!fir.heap<!fir.char<1,11>>>}>>) -> !fir.box<!fir.type<_QMtypesTt8{c:!fir.box<!fir.heap<!fir.char<1,11>>>}>>
! CHECK:           %[[VAL_14:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]*}}>>
! CHECK:           %[[VAL_15:.*]] = arith.constant {{[0-9]*}} : i32
! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_13]] : (!fir.box<!fir.type<_QMtypesTt8{c:!fir.box<!fir.heap<!fir.char<1,11>>>}>>) -> !fir.box<none>
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_14]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_18:.*]] = fir.call @_FortranAInitialize(%[[VAL_16]], %[[VAL_17]], %[[VAL_15]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:           %[[VAL_19:.*]] = arith.constant 11 : index
! CHECK:           %[[VAL_20:.*]] = hlfir.designate %[[VAL_12]]#0{"c"}   typeparams %[[VAL_19]] {fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QMtypesTt8{c:!fir.box<!fir.heap<!fir.char<1,11>>>}>>, index) -> !fir.ref<!fir.box<!fir.heap<!fir.char<1,11>>>>
! CHECK:           hlfir.assign %[[VAL_11]]#0 to %[[VAL_20]] realloc keep_lhs_len temporary_lhs : !fir.ref<!fir.char<1,12>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,11>>>>
! CHECK:           hlfir.assign %[[VAL_12]]#0 to %[[VAL_2]]#0 : !fir.ref<!fir.type<_QMtypesTt8{c:!fir.box<!fir.heap<!fir.char<1,11>>>}>>, !fir.ref<!fir.type<_QMtypesTt8{c:!fir.box<!fir.heap<!fir.char<1,11>>>}>>
! CHECK:           return
! CHECK:         }


! Test character non-allocatable component initialization
! from character allocatable of different size.
subroutine test10
  use types
  character(12), allocatable :: x
  type(t1) res
  res = t1(x)
end subroutine test10
! CHECK-LABEL:   func.func @_QPtest10() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.type<_QMtypesTt1{c:!fir.char<1,4>}>
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.type<_QMtypesTt1{c:!fir.char<1,4>}> {bindc_name = "res", uniq_name = "_QFtest10Eres"}
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFtest10Eres"} : (!fir.ref<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>) -> (!fir.ref<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>, !fir.ref<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>)
! CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,12>>> {bindc_name = "x", uniq_name = "_QFtest10Ex"}
! CHECK:           %[[VAL_4:.*]] = arith.constant 12 : index
! CHECK:           %[[VAL_5:.*]] = fir.zero_bits !fir.heap<!fir.char<1,12>>
! CHECK:           %[[VAL_6:.*]] = fir.embox %[[VAL_5]] : (!fir.heap<!fir.char<1,12>>) -> !fir.box<!fir.heap<!fir.char<1,12>>>
! CHECK:           fir.store %[[VAL_6]] to %[[VAL_3]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,12>>>>
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_3]] typeparams %[[VAL_4]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest10Ex"} : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,12>>>>, index) -> (!fir.ref<!fir.box<!fir.heap<!fir.char<1,12>>>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,12>>>>)
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "ctor.temp"} : (!fir.ref<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>) -> (!fir.ref<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>, !fir.ref<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>)
! CHECK:           %[[VAL_9:.*]] = fir.embox %[[VAL_8]]#0 : (!fir.ref<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>) -> !fir.box<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>
! CHECK:           %[[VAL_10:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]*}}>>
! CHECK:           %[[VAL_11:.*]] = arith.constant {{[0-9]*}} : i32
! CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_9]] : (!fir.box<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>) -> !fir.box<none>
! CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_10]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_14:.*]] = fir.call @_FortranAInitialize(%[[VAL_12]], %[[VAL_13]], %[[VAL_11]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:           %[[VAL_15:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_16:.*]] = hlfir.designate %[[VAL_8]]#0{"c"}   typeparams %[[VAL_15]] : (!fir.ref<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>, index) -> !fir.ref<!fir.char<1,4>>
! CHECK:           %[[VAL_17:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,12>>>>
! CHECK:           %[[VAL_18:.*]] = fir.box_addr %[[VAL_17]] : (!fir.box<!fir.heap<!fir.char<1,12>>>) -> !fir.heap<!fir.char<1,12>>
! CHECK:           hlfir.assign %[[VAL_18]] to %[[VAL_16]] temporary_lhs : !fir.heap<!fir.char<1,12>>, !fir.ref<!fir.char<1,4>>
! CHECK:           hlfir.assign %[[VAL_8]]#0 to %[[VAL_2]]#0 : !fir.ref<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>, !fir.ref<!fir.type<_QMtypesTt1{c:!fir.char<1,4>}>>
! CHECK:           return
! CHECK:         }

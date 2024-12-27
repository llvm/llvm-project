! Test lowering of Cray pointee references.
! RUN: bbc -emit-hlfir -o - -I nowhere %s 2>&1 | FileCheck %s


subroutine test1()
  real x(10), res
  pointer (cp, x)
  res = x(7)
end subroutine test1
! CHECK-LABEL:   func.func @_QPtest1() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:           %[[VAL_1:.*]] = fir.alloca i64 {bindc_name = "cp", uniq_name = "_QFtest1Ecp"}
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFtest1Ecp"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:           %[[VAL_5:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest1Ex"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
! CHECK:           %[[VAL_13:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xf32>>
! CHECK:           %[[VAL_14:.*]] = fir.embox %[[VAL_13]](%[[VAL_7]]) : (!fir.ptr<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:           fir.store %[[VAL_14]] to %[[VAL_12]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i64>>
! CHECK:           %[[VAL_16:.*]] = fir.load %[[VAL_15]] : !fir.ref<!fir.ptr<i64>>
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_12]]#0 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_16]] : (!fir.ptr<i64>) -> !fir.llvm_ptr<i8>
! CHECK:           %[[VAL_19:.*]] = fir.call @_FortranAPointerAssociateScalar(%[[VAL_17]], %[[VAL_18]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> none
! CHECK:           %[[VAL_20:.*]] = fir.load %[[VAL_12]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:           %[[VAL_21:.*]] = arith.constant 7 : index
! CHECK:           %[[VAL_22:.*]] = hlfir.designate %[[VAL_20]] (%[[VAL_21]])  : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> !fir.ref<f32>


subroutine test2(n)
  integer n
  real x
  pointer (cp, x(3:n))
  res = x(7)
end subroutine test2
! CHECK-LABEL:   func.func @_QPtest2(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:           %[[VAL_2:.*]] = fir.alloca i64 {bindc_name = "cp", uniq_name = "_QFtest2Ecp"}
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFtest2Ecp"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:           %[[VAL_19:.*]] = fir.shape_shift %{{.*}}, %{{.*}} : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_24:.*]]:2 = hlfir.declare %[[VAL_1]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest2Ex"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
! CHECK:           %[[VAL_25:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xf32>>
! CHECK:           %[[VAL_26:.*]] = fir.embox %[[VAL_25]](%[[VAL_19]]) : (!fir.ptr<!fir.array<?xf32>>, !fir.shapeshift<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:           fir.store %[[VAL_26]] to %[[VAL_24]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_3]]#0 : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i64>>
! CHECK:           %[[VAL_28:.*]] = fir.load %[[VAL_27]] : !fir.ref<!fir.ptr<i64>>
! CHECK:           %[[VAL_29:.*]] = fir.convert %[[VAL_24]]#0 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_30:.*]] = fir.convert %[[VAL_28]] : (!fir.ptr<i64>) -> !fir.llvm_ptr<i8>
! CHECK:           %[[VAL_31:.*]] = fir.call @_FortranAPointerAssociateScalar(%[[VAL_29]], %[[VAL_30]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> none
! CHECK:           %[[VAL_32:.*]] = fir.load %[[VAL_24]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:           %[[VAL_33:.*]] = arith.constant 7 : index
! CHECK:           %[[VAL_34:.*]] = hlfir.designate %[[VAL_32]] (%[[VAL_33]])  : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> !fir.ref<f32>
! CHECK:           %[[VAL_35:.*]] = fir.load %[[VAL_34]] : !fir.ref<f32>

subroutine test3(cp, n)
  character(len=11) :: c(n:2*n), res
  pointer (cp, c)
  res = c(7)
end subroutine test3
! CHECK-LABEL:   func.func @_QPtest3(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<i64> {fir.bindc_name = "cp"},
! CHECK-SAME:                        %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,11>>>>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFtest3En"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFtest3Ecp"} : (!fir.ref<i64>, !fir.dscope) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:           %[[VAL_5:.*]] = arith.constant 11 : index
! CHECK:           %[[VAL_8:.*]] = arith.constant 11 : index
! CHECK:           %[[VAL_24:.*]] = fir.shape_shift %{{.*}}, %{{.*}} : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_29:.*]]:2 = hlfir.declare %[[VAL_2]] typeparams %[[VAL_8]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest3Ec"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,11>>>>>, index) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,11>>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,11>>>>>)
! CHECK:           %[[VAL_30:.*]] = fir.zero_bits !fir.ptr<!fir.array<?x!fir.char<1,11>>>
! CHECK:           %[[VAL_31:.*]] = fir.embox %[[VAL_30]](%[[VAL_24]]) : (!fir.ptr<!fir.array<?x!fir.char<1,11>>>, !fir.shapeshift<1>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,11>>>>
! CHECK:           fir.store %[[VAL_31]] to %[[VAL_29]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,11>>>>>
! CHECK:           %[[VAL_32:.*]] = fir.convert %[[VAL_4]]#0 : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i64>>
! CHECK:           %[[VAL_33:.*]] = fir.load %[[VAL_32]] : !fir.ref<!fir.ptr<i64>>
! CHECK:           %[[VAL_34:.*]] = fir.convert %[[VAL_29]]#0 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,11>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_35:.*]] = fir.convert %[[VAL_33]] : (!fir.ptr<i64>) -> !fir.llvm_ptr<i8>
! CHECK:           %[[VAL_36:.*]] = fir.call @_FortranAPointerAssociateScalar(%[[VAL_34]], %[[VAL_35]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> none
! CHECK:           %[[VAL_37:.*]] = fir.load %[[VAL_29]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,11>>>>>
! CHECK:           %[[VAL_38:.*]] = arith.constant 7 : index
! CHECK:           %[[VAL_39:.*]] = hlfir.designate %[[VAL_37]] (%[[VAL_38]])  typeparams %[[VAL_8]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,11>>>>, index, index) -> !fir.ref<!fir.char<1,11>>

subroutine test4(n)
  character(len=n) :: c, res
  pointer (cp, c)
  res = c
end subroutine test4
! CHECK-LABEL:   func.func @_QPtest4(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.char<1,?>>>
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFtest4En"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_3:.*]] = fir.alloca i64 {bindc_name = "cp", uniq_name = "_QFtest4Ecp"}
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] {uniq_name = "_QFtest4Ecp"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_6:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_7:.*]] = arith.cmpi sgt, %[[VAL_5]], %[[VAL_6]] : i32
! CHECK:           %[[VAL_8:.*]] = arith.select %[[VAL_7]], %[[VAL_5]], %[[VAL_6]] : i32
! CHECK:           %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_1]] typeparams %[[VAL_8]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest4Ec"} : (!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>, i32) -> (!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>)
! CHECK:           %[[VAL_14:.*]] = fir.zero_bits !fir.ptr<!fir.char<1,?>>
! CHECK:           %[[VAL_15:.*]] = fir.embox %[[VAL_14]] typeparams %[[VAL_8]] : (!fir.ptr<!fir.char<1,?>>, i32) -> !fir.box<!fir.ptr<!fir.char<1,?>>>
! CHECK:           fir.store %[[VAL_15]] to %[[VAL_13]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
! CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_4]]#0 : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i64>>
! CHECK:           %[[VAL_23:.*]] = fir.load %[[VAL_22]] : !fir.ref<!fir.ptr<i64>>
! CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_13]]#0 : (!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_23]] : (!fir.ptr<i64>) -> !fir.llvm_ptr<i8>
! CHECK:           %[[VAL_26:.*]] = fir.call @_FortranAPointerAssociateScalar(%[[VAL_24]], %[[VAL_25]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> none
! CHECK:           %[[VAL_27:.*]] = fir.load %[[VAL_13]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
! CHECK:           %[[VAL_28:.*]] = fir.box_addr %[[VAL_27]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> !fir.ptr<!fir.char<1,?>>
! CHECK:           %[[VAL_29:.*]] = fir.emboxchar %[[VAL_28]], %[[VAL_8]] : (!fir.ptr<!fir.char<1,?>>, i32) -> !fir.boxchar<1>

subroutine test5()
  type t
     sequence
     real r
     integer i
  end type t
  type(t) :: v
  integer res
  pointer (cp, v(3:11))
  res = v(7)%i
end subroutine test5
! CHECK-LABEL:   func.func @_QPtest5() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFtest5Tt{r:f32,i:i32}>>>>
! CHECK:           %[[VAL_1:.*]] = fir.alloca i64 {bindc_name = "cp", uniq_name = "_QFtest5Ecp"}
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFtest5Ecp"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:           %[[VAL_5:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_6:.*]] = arith.constant 9 : index
! CHECK:           %[[VAL_8:.*]] = fir.shape_shift %[[VAL_5]], %[[VAL_6]] : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest5Ev"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFtest5Tt{r:f32,i:i32}>>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFtest5Tt{r:f32,i:i32}>>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFtest5Tt{r:f32,i:i32}>>>>>)
! CHECK:           %[[VAL_14:.*]] = fir.zero_bits !fir.ptr<!fir.array<?x!fir.type<_QFtest5Tt{r:f32,i:i32}>>>
! CHECK:           %[[VAL_15:.*]] = fir.embox %[[VAL_14]](%[[VAL_8]]) : (!fir.ptr<!fir.array<?x!fir.type<_QFtest5Tt{r:f32,i:i32}>>>, !fir.shapeshift<1>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFtest5Tt{r:f32,i:i32}>>>>
! CHECK:           fir.store %[[VAL_15]] to %[[VAL_13]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFtest5Tt{r:f32,i:i32}>>>>>
! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i64>>
! CHECK:           %[[VAL_17:.*]] = fir.load %[[VAL_16]] : !fir.ref<!fir.ptr<i64>>
! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_13]]#0 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFtest5Tt{r:f32,i:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_17]] : (!fir.ptr<i64>) -> !fir.llvm_ptr<i8>
! CHECK:           %[[VAL_20:.*]] = fir.call @_FortranAPointerAssociateScalar(%[[VAL_18]], %[[VAL_19]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> none
! CHECK:           %[[VAL_21:.*]] = fir.load %[[VAL_13]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFtest5Tt{r:f32,i:i32}>>>>>
! CHECK:           %[[VAL_22:.*]] = arith.constant 7 : index
! CHECK:           %[[VAL_23:.*]] = hlfir.designate %[[VAL_21]] (%[[VAL_22]])  : (!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFtest5Tt{r:f32,i:i32}>>>>, index) -> !fir.ref<!fir.type<_QFtest5Tt{r:f32,i:i32}>>
! CHECK:           %[[VAL_24:.*]] = hlfir.designate %[[VAL_23]]{"i"}   : (!fir.ref<!fir.type<_QFtest5Tt{r:f32,i:i32}>>) -> !fir.ref<i32>
! CHECK:           %[[VAL_25:.*]] = fir.load %[[VAL_24]] : !fir.ref<i32>

subroutine test6(n)
  character(len=n) :: c(20), res1
  real x, res2
  pointer (cp, c)
  pointer (cp, x(n:17))
  res1 = c(9)
  res2 = x(5)
end subroutine test6
! CHECK-LABEL:   func.func @_QPtest6(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFtest6En"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_4:.*]] = fir.alloca i64 {bindc_name = "cp", uniq_name = "_QFtest6Ecp"}
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] {uniq_name = "_QFtest6Ecp"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_9:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_10:.*]] = arith.cmpi sgt, %[[VAL_8]], %[[VAL_9]] : i32
! CHECK:           %[[VAL_11:.*]] = arith.select %[[VAL_10]], %[[VAL_8]], %[[VAL_9]] : i32
! CHECK:           %[[VAL_12:.*]] = arith.constant 20 : index

! CHECK:           %[[VAL_14:.*]] = fir.shape %[[VAL_12]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_20:.*]]:2 = hlfir.declare %[[VAL_2]] typeparams %[[VAL_11]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest6Ec"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>, i32) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>)
! CHECK:           %[[VAL_21:.*]] = fir.zero_bits !fir.ptr<!fir.array<?x!fir.char<1,?>>>
! CHECK:           %[[VAL_22:.*]] = fir.embox %[[VAL_21]](%[[VAL_14]]) typeparams %[[VAL_11]] : (!fir.ptr<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, i32) -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
! CHECK:           fir.store %[[VAL_22]] to %[[VAL_20]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>

! CHECK:           %[[VAL_41:.*]] = fir.shape_shift %{{.*}}, %{{.*}} : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_46:.*]]:2 = hlfir.declare %[[VAL_1]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest6Ex"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
! CHECK:           %[[VAL_47:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xf32>>
! CHECK:           %[[VAL_48:.*]] = fir.embox %[[VAL_47]](%[[VAL_41]]) : (!fir.ptr<!fir.array<?xf32>>, !fir.shapeshift<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:           fir.store %[[VAL_48]] to %[[VAL_46]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>

! CHECK:           %[[VAL_49:.*]] = fir.convert %[[VAL_5]]#0 : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i64>>
! CHECK:           %[[VAL_50:.*]] = fir.load %[[VAL_49]] : !fir.ref<!fir.ptr<i64>>
! CHECK:           %[[VAL_51:.*]] = fir.convert %[[VAL_20]]#0 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_52:.*]] = fir.convert %[[VAL_50]] : (!fir.ptr<i64>) -> !fir.llvm_ptr<i8>
! CHECK:           %[[VAL_53:.*]] = fir.call @_FortranAPointerAssociateScalar(%[[VAL_51]], %[[VAL_52]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> none
! CHECK:           %[[VAL_54:.*]] = fir.load %[[VAL_20]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>
! CHECK:           %[[VAL_55:.*]] = arith.constant 9 : index
! CHECK:           %[[VAL_56:.*]] = hlfir.designate %[[VAL_54]] (%[[VAL_55]])  typeparams %[[VAL_11]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>, index, i32) -> !fir.boxchar<1>

! CHECK:           %[[VAL_57:.*]] = fir.convert %[[VAL_5]]#0 : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i64>>
! CHECK:           %[[VAL_58:.*]] = fir.load %[[VAL_57]] : !fir.ref<!fir.ptr<i64>>
! CHECK:           %[[VAL_59:.*]] = fir.convert %[[VAL_46]]#0 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_60:.*]] = fir.convert %[[VAL_58]] : (!fir.ptr<i64>) -> !fir.llvm_ptr<i8>
! CHECK:           %[[VAL_61:.*]] = fir.call @_FortranAPointerAssociateScalar(%[[VAL_59]], %[[VAL_60]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> none
! CHECK:           %[[VAL_62:.*]] = fir.load %[[VAL_46]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:           %[[VAL_63:.*]] = arith.constant 5 : index
! CHECK:           %[[VAL_64:.*]] = hlfir.designate %[[VAL_62]] (%[[VAL_63]])  : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> !fir.ref<f32>
! CHECK:           %[[VAL_65:.*]] = fir.load %[[VAL_64]] : !fir.ref<f32>

subroutine test7()
  integer :: pte, arr(5)
  pointer(ptr, pte(5))
  arr = pte
end subroutine test7
! CHECK-LABEL:     func.func @_QPtest7(
! CHECK:    %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:    %[[VAL_2:.*]] = arith.constant 5 : index
! CHECK:    %[[VAL_3:.*]] = fir.alloca !fir.array<5xi32> {bindc_name = "arr", uniq_name = "_QFtest7Earr"}
! CHECK:    %[[VAL_4:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:    %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_3]](%[[VAL_4]]) {uniq_name = "_QFtest7Earr"} : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<5xi32>>, !fir.ref<!fir.array<5xi32>>)
! CHECK:    %[[VAL_12:.*]] = fir.alloca i64 {bindc_name = "ptr", uniq_name = "_QFtest7Eptr"}
! CHECK:    %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_12]] {uniq_name = "_QFtest7Eptr"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:    %[[VAL_6:.*]] = arith.constant 5 : index
! CHECK:    %[[VAL_8:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:    %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_1]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest7Epte"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
! CHECK:    %[[VAL_10:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
! CHECK:    %[[VAL_11:.*]] = fir.embox %[[VAL_10]](%[[VAL_8]]) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:    fir.store %[[VAL_11]] to %[[VAL_9]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:    %[[VAL_14:.*]] = fir.convert %[[VAL_13]]#0 : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i64>>
! CHECK:    %[[VAL_15:.*]] = fir.load %[[VAL_14]] : !fir.ref<!fir.ptr<i64>>
! CHECK:    %[[VAL_16:.*]] = fir.convert %[[VAL_9]]#0 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:    %[[VAL_17:.*]] = fir.convert %[[VAL_15]] : (!fir.ptr<i64>) -> !fir.llvm_ptr<i8>
! CHECK:    %[[VAL_18:.*]] = fir.call @_FortranAPointerAssociateScalar(%[[VAL_16]], %[[VAL_17]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> none
! CHECK:    %[[VAL_19:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>

subroutine test8()
  integer :: pte
  pointer(ptr, pte(5))
  call sub(pte)
end subroutine test8
! CHECK-LABEL:     func.func @_QPtest8(
! CHECK:    %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:    %[[VAL_8:.*]] = fir.alloca i64 {bindc_name = "ptr", uniq_name = "_QFtest8Eptr"}
! CHECK:    %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]] {uniq_name = "_QFtest8Eptr"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:    %[[VAL_2:.*]] = arith.constant 5 : index
! CHECK:    %[[VAL_4:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:    %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_1]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest8Epte"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
! CHECK:    %[[VAL_6:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
! CHECK:    %[[VAL_7:.*]] = fir.embox %[[VAL_6]](%[[VAL_4]]) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:    fir.store %[[VAL_7]] to %[[VAL_5]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:    %[[VAL_10:.*]] = fir.convert %[[VAL_9]]#0 : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i64>>
! CHECK:    %[[VAL_11:.*]] = fir.load %[[VAL_10]] : !fir.ref<!fir.ptr<i64>>
! CHECK:    %[[VAL_12:.*]] = fir.convert %[[VAL_5]]#0 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:    %[[VAL_13:.*]] = fir.convert %[[VAL_11]] : (!fir.ptr<i64>) -> !fir.llvm_ptr<i8>
! CHECK:    %[[VAL_14:.*]] = fir.call @_FortranAPointerAssociateScalar(%[[VAL_12]], %[[VAL_13]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> none
! CHECK:    %[[VAL_15:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:    %[[VAL_16:.*]] = fir.box_addr %[[VAL_15]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ptr<!fir.array<?xi32>>
! CHECK:    %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (!fir.ptr<!fir.array<?xi32>>) -> !fir.ref<!fir.array<5xi32>>
! CHECK:    fir.call @_QPsub(%[[VAL_17]]) fastmath<contract> : (!fir.ref<!fir.array<5xi32>>) -> ()

subroutine test9()
  integer :: pte
  pointer(ptr, pte(5))
  interface
    subroutine sub(x)
    integer, value :: x(5)
    end
  end interface
  call sub(pte)
end subroutine test9
! CHECK-LABEL:     func.func @_QPtest9(
! CHECK:    %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:    %[[VAL_8:.*]] = fir.alloca i64 {bindc_name = "ptr", uniq_name = "_QFtest9Eptr"}
! CHECK:    %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]] {uniq_name = "_QFtest9Eptr"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:    %[[VAL_2:.*]] = arith.constant 5 : index
! CHECK:    %[[VAL_4:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:    %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_1]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest9Epte"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
! CHECK:    %[[VAL_6:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
! CHECK:    %[[VAL_7:.*]] = fir.embox %[[VAL_6]](%[[VAL_4]]) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:    fir.store %[[VAL_7]] to %[[VAL_5]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:    %[[VAL_10:.*]] = fir.convert %[[VAL_9]]#0 : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i64>>
! CHECK:    %[[VAL_11:.*]] = fir.load %[[VAL_10]] : !fir.ref<!fir.ptr<i64>>
! CHECK:    %[[VAL_12:.*]] = fir.convert %[[VAL_5]]#0 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:    %[[VAL_13:.*]] = fir.convert %[[VAL_11]] : (!fir.ptr<i64>) -> !fir.llvm_ptr<i8>
! CHECK:    %[[VAL_14:.*]] = fir.call @_FortranAPointerAssociateScalar(%[[VAL_12]], %[[VAL_13]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> none
! CHECK:    %[[VAL_15:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:    %[[VAL_16:.*]] = hlfir.as_expr %[[VAL_15]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !hlfir.expr<?xi32>
! CHECK:    %[[VAL_17:.*]] = arith.constant 0 : index
! CHECK:    %[[VAL_18:.*]]:3 = fir.box_dims %[[VAL_15]], %[[VAL_17]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:    %[[VAL_19:.*]] = fir.shape %[[VAL_18]]#1 : (index) -> !fir.shape<1>
! CHECK:    %[[VAL_20:.*]]:3 = hlfir.associate %[[VAL_16]](%[[VAL_19]]) {adapt.valuebyref} : (!hlfir.expr<?xi32>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>, i1)
! CHECK:    %[[VAL_21:.*]] = fir.convert %[[VAL_20]]#1 : (!fir.ref<!fir.array<?xi32>>) -> !fir.ref<!fir.array<5xi32>>
! CHECK:    fir.call @_QPsub(%[[VAL_21]]) fastmath<contract> : (!fir.ref<!fir.array<5xi32>>) -> ()
! CHECK:    hlfir.end_associate %[[VAL_20]]#1, %[[VAL_20]]#2 : !fir.ref<!fir.array<?xi32>>, i1


subroutine test10()
  integer :: pte
  pointer(ptr, pte)
  call sub1(pte)
end subroutine test10
! CHECK-LABEL:  func.func @_QPtest10(
! CHECK:    %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.ptr<i32>>
! CHECK:    %[[VAL_6:.*]] = fir.alloca i64 {bindc_name = "ptr", uniq_name = "_QFtest10Eptr"}
! CHECK:    %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]] {uniq_name = "_QFtest10Eptr"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:    %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest10Epte"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
! CHECK:    %[[VAL_4:.*]] = fir.zero_bits !fir.ptr<i32>
! CHECK:    %[[VAL_5:.*]] = fir.embox %[[VAL_4]] : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
! CHECK:    fir.store %[[VAL_5]] to %[[VAL_3]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:    %[[VAL_8:.*]] = fir.convert %[[VAL_7]]#0 : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i64>>
! CHECK:    %[[VAL_9:.*]] = fir.load %[[VAL_8]] : !fir.ref<!fir.ptr<i64>>
! CHECK:    %[[VAL_10:.*]] = fir.convert %[[VAL_3]]#0 : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.ref<!fir.box<none>>
! CHECK:    %[[VAL_11:.*]] = fir.convert %[[VAL_9]] : (!fir.ptr<i64>) -> !fir.llvm_ptr<i8>
! CHECK:    %[[VAL_12:.*]] = fir.call @_FortranAPointerAssociateScalar(%[[VAL_10]], %[[VAL_11]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> none
! CHECK:    %[[VAL_13:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:    %[[VAL_14:.*]] = fir.box_addr %[[VAL_13]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:    %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (!fir.ptr<i32>) -> !fir.ref<i32>
! CHECK:    fir.call @_QPsub1(%[[VAL_15]]) fastmath<contract> : (!fir.ref<i32>) -> ()

subroutine test11()
  integer :: pte
  pointer(ptr, pte)
  interface
    subroutine sub2(x)
    integer, value :: x
    end
  end interface
  call sub2(pte)
end subroutine test11
! CHECK-LABEL:  func.func @_QPtest11(
! CHECK:    %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.ptr<i32>>
! CHECK:    %[[VAL_6:.*]] = fir.alloca i64 {bindc_name = "ptr", uniq_name = "_QFtest11Eptr"}
! CHECK:    %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]] {uniq_name = "_QFtest11Eptr"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:    %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest11Epte"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
! CHECK:    %[[VAL_4:.*]] = fir.zero_bits !fir.ptr<i32>
! CHECK:    %[[VAL_5:.*]] = fir.embox %[[VAL_4]] : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
! CHECK:    fir.store %[[VAL_5]] to %[[VAL_3]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:    %[[VAL_8:.*]] = fir.convert %[[VAL_7]]#0 : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i64>>
! CHECK:    %[[VAL_9:.*]] = fir.load %[[VAL_8]] : !fir.ref<!fir.ptr<i64>>
! CHECK:    %[[VAL_10:.*]] = fir.convert %[[VAL_3]]#0 : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.ref<!fir.box<none>>
! CHECK:    %[[VAL_11:.*]] = fir.convert %[[VAL_9]] : (!fir.ptr<i64>) -> !fir.llvm_ptr<i8>
! CHECK:    %[[VAL_12:.*]] = fir.call @_FortranAPointerAssociateScalar(%[[VAL_10]], %[[VAL_11]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> none
! CHECK:    %[[VAL_13:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:    %[[VAL_14:.*]] = fir.box_addr %[[VAL_13]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:    %[[VAL_15:.*]] = fir.load %[[VAL_14]] : !fir.ptr<i32>
! CHECK:    fir.call @_QPsub2(%[[VAL_15]]) fastmath<contract> : (i32) -> ()

module test_mod
  integer(8) :: cray_pointer
  real :: cray_pointee
  pointer(cray_pointer, cray_pointee)
end module

subroutine test_hidden_pointer
  ! Only the pointee is accessed, yet the pointer is needed
  ! for lowering.
  use test_mod, only : cray_pointee
  call takes_real(cray_pointee)
end
! CHECK-LABEL:   func.func @_QPtest_hidden_pointer() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.ptr<f32>>
! CHECK:           %[[VAL_1:.*]] = fir.address_of(@_QMtest_modEcray_pointer) : !fir.ref<i64>
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QMtest_modEcray_pointer"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMtest_modEcray_pointee"} : (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> (!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.ptr<f32>>>)
! CHECK:           %[[VAL_4:.*]] = fir.zero_bits !fir.ptr<f32>
! CHECK:           %[[VAL_5:.*]] = fir.embox %[[VAL_4]] : (!fir.ptr<f32>) -> !fir.box<!fir.ptr<f32>>
! CHECK:           fir.store %[[VAL_5]] to %[[VAL_3]]#0 : !fir.ref<!fir.box<!fir.ptr<f32>>>
! CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i64>>
! CHECK:           %[[VAL_7:.*]] = fir.load %[[VAL_6]] : !fir.ref<!fir.ptr<i64>>
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_3]]#0 : (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_7]] : (!fir.ptr<i64>) -> !fir.llvm_ptr<i8>
! CHECK:           %[[VAL_10:.*]] = fir.call @_FortranAPointerAssociateScalar(%[[VAL_8]], %[[VAL_9]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> none
! CHECK:           %[[VAL_11:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.box<!fir.ptr<f32>>>
! CHECK:           %[[VAL_12:.*]] = fir.box_addr %[[VAL_11]] : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
! CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (!fir.ptr<f32>) -> !fir.ref<f32>
! CHECK:           fir.call @_QPtakes_real(%[[VAL_13]]) fastmath<contract> : (!fir.ref<f32>) -> ()
! CHECK:           return
! CHECK:         }



subroutine test_craypointer_capture(n)
  integer :: n
  character(n) :: cray_pointee
  integer(8) :: cray_pointer
  pointer(cray_pointer, cray_pointee)
  call internal()
  contains
subroutine internal()
  call takes_character(cray_pointee)
end subroutine
end subroutine
! CHECK-LABEL:   func.func @_QPtest_craypointer_capture(
! CHECK-SAME:                                           %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.char<1,?>>>
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFtest_craypointer_captureEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_3:.*]] = fir.alloca i64 {bindc_name = "cray_pointer", uniq_name = "_QFtest_craypointer_captureEcray_pointer"}
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] {fortran_attrs = #fir.var_attrs<internal_assoc>, uniq_name = "_QFtest_craypointer_captureEcray_pointer"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_6:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_7:.*]] = arith.cmpi sgt, %[[VAL_5]], %[[VAL_6]] : i32
! CHECK:           %[[VAL_8:.*]] = arith.select %[[VAL_7]], %[[VAL_5]], %[[VAL_6]] : i32
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_1]] typeparams %[[VAL_8]] {fortran_attrs = #fir.var_attrs<pointer, internal_assoc>, uniq_name = "_QFtest_craypointer_captureEcray_pointee"} : (!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>, i32) -> (!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>)
! CHECK:           %[[VAL_10:.*]] = fir.zero_bits !fir.ptr<!fir.char<1,?>>
! CHECK:           %[[VAL_11:.*]] = fir.embox %[[VAL_10]] typeparams %[[VAL_8]] : (!fir.ptr<!fir.char<1,?>>, i32) -> !fir.box<!fir.ptr<!fir.char<1,?>>>
! CHECK:           fir.store %[[VAL_11]] to %[[VAL_9]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
! CHECK:           %[[VAL_12:.*]] = fir.alloca tuple<!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>, !fir.ref<i64>>
! CHECK:           %[[VAL_13:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_14:.*]] = fir.coordinate_of %[[VAL_12]], %[[VAL_13]] : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>, !fir.ref<i64>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>>
! CHECK:           fir.store %[[VAL_9]]#1 to %[[VAL_14]] : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>>
! CHECK:           %[[VAL_15:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_16:.*]] = fir.coordinate_of %[[VAL_12]], %[[VAL_15]] : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>, !fir.ref<i64>>>, i32) -> !fir.llvm_ptr<!fir.ref<i64>>
! CHECK:           fir.store %[[VAL_4]]#1 to %[[VAL_16]] : !fir.llvm_ptr<!fir.ref<i64>>
! CHECK:           fir.call @_QFtest_craypointer_capturePinternal(%[[VAL_12]]) fastmath<contract> : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>, !fir.ref<i64>>>) -> ()
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func private @_QFtest_craypointer_capturePinternal(
! CHECK-SAME:                                                            %[[VAL_0:.*]]: !fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>, !fir.ref<i64>>> {fir.host_assoc})
! CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_2:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_1]] : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>, !fir.ref<i64>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>>
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_2]] : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>>
! CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
! CHECK:           %[[VAL_5:.*]] = fir.box_elesize %[[VAL_4]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> index
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_3]] typeparams %[[VAL_5]] {fortran_attrs = #fir.var_attrs<pointer, host_assoc>, uniq_name = "_QFtest_craypointer_captureEcray_pointee"} : (!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>, index) -> (!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>)
! CHECK:           %[[VAL_7:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_8:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_7]] : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>, !fir.ref<i64>>>, i32) -> !fir.llvm_ptr<!fir.ref<i64>>
! CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_8]] : !fir.llvm_ptr<!fir.ref<i64>>
! CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_9]] {fortran_attrs = #fir.var_attrs<host_assoc>, uniq_name = "_QFtest_craypointer_captureEcray_pointer"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_10]]#0 : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i64>>
! CHECK:           %[[VAL_12:.*]] = fir.load %[[VAL_11]] : !fir.ref<!fir.ptr<i64>>
! CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_6]]#0 : (!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_12]] : (!fir.ptr<i64>) -> !fir.llvm_ptr<i8>
! CHECK:           %[[VAL_15:.*]] = fir.call @_FortranAPointerAssociateScalar(%[[VAL_13]], %[[VAL_14]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> none
! CHECK:           %[[VAL_16:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
! CHECK:           %[[VAL_17:.*]] = fir.box_addr %[[VAL_16]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> !fir.ptr<!fir.char<1,?>>
! CHECK:           %[[VAL_18:.*]] = fir.emboxchar %[[VAL_17]], %[[VAL_5]] : (!fir.ptr<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:           fir.call @_QPtakes_character(%[[VAL_18]]) fastmath<contract> : (!fir.boxchar<1>) -> ()
! CHECK:           return
! CHECK:         }

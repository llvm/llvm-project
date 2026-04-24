! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! Test lowering of local character variables

! CHECK-LABEL: func @_QPscalar_cst_len
subroutine scalar_cst_len()
  character(10) :: c
  ! CHECK: %[[VAL_0:.*]] = arith.constant 10 : index
  ! CHECK: %[[VAL_1:.*]] = fir.alloca !fir.char<1,10> {bindc_name = "c", uniq_name = "_QFscalar_cst_lenEc"}
  ! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] typeparams %[[VAL_0]] {uniq_name = "_QFscalar_cst_lenEc"} : (!fir.ref<!fir.char<1,10>>, index) -> (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>)
  print *, c
end subroutine

! CHECK-LABEL: func @_QPscalar_dyn_len
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<i32>
subroutine scalar_dyn_len(l)
  integer :: l
  character(l) :: c
  ! CHECK: %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] arg 1 {uniq_name = "_QFscalar_dyn_lenEl"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK: %[[VAL_2:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<i32>
  ! CHECK: %[[VAL_3:.*]] = arith.constant 0 : i32
  ! CHECK: %[[VAL_4:.*]] = arith.cmpi sgt, %[[VAL_2]], %[[VAL_3]] : i32
  ! CHECK: %[[VAL_5:.*]] = arith.select %[[VAL_4]], %[[VAL_2]], %[[VAL_3]] : i32
  ! CHECK: %[[VAL_6:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_5]] : i32) {bindc_name = "c", uniq_name = "_QFscalar_dyn_lenEc"}
  ! CHECK: %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]] typeparams %[[VAL_5]] {uniq_name = "_QFscalar_dyn_lenEc"} : (!fir.ref<!fir.char<1,?>>, i32) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  print *, c
end subroutine

! CHECK-LABEL: func @_QPcst_array_cst_len
subroutine cst_array_cst_len()
  character(10) :: c(20)
  ! CHECK: %[[VAL_0:.*]] = arith.constant 10 : index
  ! CHECK: %[[VAL_1:.*]] = arith.constant 20 : index
  ! CHECK: %[[VAL_2:.*]] = fir.alloca !fir.array<20x!fir.char<1,10>> {bindc_name = "c", uniq_name = "_QFcst_array_cst_lenEc"}
  ! CHECK: %[[VAL_3:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_2]](%[[VAL_3]]) typeparams %[[VAL_0]] {uniq_name = "_QFcst_array_cst_lenEc"} : (!fir.ref<!fir.array<20x!fir.char<1,10>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<20x!fir.char<1,10>>>, !fir.ref<!fir.array<20x!fir.char<1,10>>>)
  print *, c(1)
end subroutine

! CHECK-LABEL: func @_QPcst_array_dyn_len
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<i32>
subroutine cst_array_dyn_len(l)
  integer :: l
  character(l) :: c(10)
  ! CHECK: %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] arg 1 {uniq_name = "_QFcst_array_dyn_lenEl"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK: %[[VAL_2:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<i32>
  ! CHECK: %[[VAL_3:.*]] = arith.constant 0 : i32
  ! CHECK: %[[VAL_4:.*]] = arith.cmpi sgt, %[[VAL_2]], %[[VAL_3]] : i32
  ! CHECK: %[[VAL_5:.*]] = arith.select %[[VAL_4]], %[[VAL_2]], %[[VAL_3]] : i32
  ! CHECK: %[[VAL_6:.*]] = arith.constant 10 : index
  ! CHECK: %[[VAL_7:.*]] = fir.alloca !fir.array<10x!fir.char<1,?>>(%[[VAL_5]] : i32) {bindc_name = "c", uniq_name = "_QFcst_array_dyn_lenEc"}
  ! CHECK: %[[VAL_8:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_7]](%[[VAL_8]]) typeparams %[[VAL_5]] {uniq_name = "_QFcst_array_dyn_lenEc"} : (!fir.ref<!fir.array<10x!fir.char<1,?>>>, !fir.shape<1>, i32) -> (!fir.box<!fir.array<10x!fir.char<1,?>>>, !fir.ref<!fir.array<10x!fir.char<1,?>>>)
  print *, c(1)
end subroutine

! CHECK-LABEL: func @_QPdyn_array_cst_len
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<i32>
subroutine dyn_array_cst_len(n)
  integer :: n
  character(10) :: c(n)
  ! CHECK: %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] arg 1 {uniq_name = "_QFdyn_array_cst_lenEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK: %[[VAL_2:.*]] = arith.constant 10 : index
  ! CHECK: %[[VAL_3:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<i32>
  ! CHECK: %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (i32) -> i64
  ! CHECK: %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i64) -> index
  ! CHECK: %[[VAL_6:.*]] = arith.constant 0 : index
  ! CHECK: %[[VAL_7:.*]] = arith.cmpi sgt, %[[VAL_5]], %[[VAL_6]] : index
  ! CHECK: %[[VAL_8:.*]] = arith.select %[[VAL_7]], %[[VAL_5]], %[[VAL_6]] : index
  ! CHECK: %[[VAL_9:.*]] = fir.alloca !fir.array<?x!fir.char<1,10>>, %[[VAL_8]] {bindc_name = "c", uniq_name = "_QFdyn_array_cst_lenEc"}
  ! CHECK: %[[VAL_10:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_9]](%[[VAL_10]]) typeparams %[[VAL_2]] {uniq_name = "_QFdyn_array_cst_lenEc"} : (!fir.ref<!fir.array<?x!fir.char<1,10>>>, !fir.shape<1>, index) -> (!fir.box<!fir.array<?x!fir.char<1,10>>>, !fir.ref<!fir.array<?x!fir.char<1,10>>>)
  print *, c(1)
end subroutine

! CHECK-LABEL: func @_QPdyn_array_dyn_len
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<i32>{{.*}}, %[[ARG1:.*]]: !fir.ref<i32>
subroutine dyn_array_dyn_len(l, n)
  integer :: l, n
  character(l) :: c(n)
  ! CHECK: %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] arg 1 {uniq_name = "_QFdyn_array_dyn_lenEl"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_0]] arg 2 {uniq_name = "_QFdyn_array_dyn_lenEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK: %[[VAL_3:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<i32>
  ! CHECK: %[[VAL_4:.*]] = arith.constant 0 : i32
  ! CHECK: %[[VAL_5:.*]] = arith.cmpi sgt, %[[VAL_3]], %[[VAL_4]] : i32
  ! CHECK: %[[VAL_6:.*]] = arith.select %[[VAL_5]], %[[VAL_3]], %[[VAL_4]] : i32
  ! CHECK: %[[VAL_7:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<i32>
  ! CHECK: %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> i64
  ! CHECK: %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i64) -> index
  ! CHECK: %[[VAL_10:.*]] = arith.constant 0 : index
  ! CHECK: %[[VAL_11:.*]] = arith.cmpi sgt, %[[VAL_9]], %[[VAL_10]] : index
  ! CHECK: %[[VAL_12:.*]] = arith.select %[[VAL_11]], %[[VAL_9]], %[[VAL_10]] : index
  ! CHECK: %[[VAL_13:.*]] = fir.alloca !fir.array<?x!fir.char<1,?>>(%[[VAL_6]] : i32), %[[VAL_12]] {bindc_name = "c", uniq_name = "_QFdyn_array_dyn_lenEc"}
  ! CHECK: %[[VAL_14:.*]] = fir.shape %[[VAL_12]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_13]](%[[VAL_14]]) typeparams %[[VAL_6]] {uniq_name = "_QFdyn_array_dyn_lenEc"} : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, i32) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.ref<!fir.array<?x!fir.char<1,?>>>)
  print *, c(1)
end subroutine

! CHECK-LABEL: func @_QPcst_array_cst_len_lb
subroutine cst_array_cst_len_lb()
  character(10) :: c(11:30)
  ! CHECK: %[[VAL_0:.*]] = arith.constant 10 : index
  ! CHECK: %[[VAL_1:.*]] = arith.constant 11 : index
  ! CHECK: %[[VAL_2:.*]] = arith.constant 20 : index
  ! CHECK: %[[VAL_3:.*]] = fir.alloca !fir.array<20x!fir.char<1,10>> {bindc_name = "c", uniq_name = "_QFcst_array_cst_len_lbEc"}
  ! CHECK: %[[VAL_4:.*]] = fir.shape_shift %[[VAL_1]], %[[VAL_2]] : (index, index) -> !fir.shapeshift<1>
  ! CHECK: %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_3]](%[[VAL_4]]) typeparams %[[VAL_0]] {uniq_name = "_QFcst_array_cst_len_lbEc"} : (!fir.ref<!fir.array<20x!fir.char<1,10>>>, !fir.shapeshift<1>, index) -> (!fir.box<!fir.array<20x!fir.char<1,10>>>, !fir.ref<!fir.array<20x!fir.char<1,10>>>)
  print *, c(11)
end subroutine

! CHECK-LABEL: func @_QPcst_array_dyn_len_lb
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<i64>
subroutine cst_array_dyn_len_lb(l)
  integer(8) :: l
  character(l) :: c(11:20)
  ! CHECK: %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] arg 1 {uniq_name = "_QFcst_array_dyn_len_lbEl"} : (!fir.ref<i64>, !fir.dscope) -> (!fir.ref<i64>, !fir.ref<i64>)
  ! CHECK: %[[VAL_2:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<i64>
  ! CHECK: %[[VAL_3:.*]] = arith.constant 0 : i64
  ! CHECK: %[[VAL_4:.*]] = arith.cmpi sgt, %[[VAL_2]], %[[VAL_3]] : i64
  ! CHECK: %[[VAL_5:.*]] = arith.select %[[VAL_4]], %[[VAL_2]], %[[VAL_3]] : i64
  ! CHECK: %[[VAL_6:.*]] = arith.constant 11 : index
  ! CHECK: %[[VAL_7:.*]] = arith.constant 10 : index
  ! CHECK: %[[VAL_8:.*]] = fir.alloca !fir.array<10x!fir.char<1,?>>(%[[VAL_5]] : i64) {bindc_name = "c", uniq_name = "_QFcst_array_dyn_len_lbEc"}
  ! CHECK: %[[VAL_9:.*]] = fir.shape_shift %[[VAL_6]], %[[VAL_7]] : (index, index) -> !fir.shapeshift<1>
  ! CHECK: %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_8]](%[[VAL_9]]) typeparams %[[VAL_5]] {uniq_name = "_QFcst_array_dyn_len_lbEc"} : (!fir.ref<!fir.array<10x!fir.char<1,?>>>, !fir.shapeshift<1>, i64) -> (!fir.box<!fir.array<10x!fir.char<1,?>>>, !fir.ref<!fir.array<10x!fir.char<1,?>>>)
  print *, c(11)
end subroutine

! CHECK-LABEL: func @_QPdyn_array_cst_len_lb
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<i64>
subroutine dyn_array_cst_len_lb(n)
  integer(8) :: n
  character(10) :: c(11:n)
  ! CHECK: %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] arg 1 {uniq_name = "_QFdyn_array_cst_len_lbEn"} : (!fir.ref<i64>, !fir.dscope) -> (!fir.ref<i64>, !fir.ref<i64>)
  ! CHECK: %[[VAL_2:.*]] = arith.constant 10 : index
  ! CHECK: %[[VAL_3:.*]] = arith.constant 11 : i64
  ! CHECK: %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (i64) -> index
  ! CHECK: %[[VAL_5:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<i64>
  ! CHECK: %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i64) -> index
  ! CHECK: %[[VAL_7:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_8:.*]] = arith.constant 0 : index
  ! CHECK: %[[VAL_9:.*]] = arith.subi %[[VAL_6]], %[[VAL_4]] : index
  ! CHECK: %[[VAL_10:.*]] = arith.addi %[[VAL_9]], %[[VAL_7]] : index
  ! CHECK: %[[VAL_11:.*]] = arith.cmpi sgt, %[[VAL_10]], %[[VAL_8]] : index
  ! CHECK: %[[VAL_12:.*]] = arith.select %[[VAL_11]], %[[VAL_10]], %[[VAL_8]] : index
  ! CHECK: %[[VAL_13:.*]] = fir.alloca !fir.array<?x!fir.char<1,10>>, %[[VAL_12]] {bindc_name = "c", uniq_name = "_QFdyn_array_cst_len_lbEc"}
  ! CHECK: %[[VAL_14:.*]] = fir.shape_shift %[[VAL_4]], %[[VAL_12]] : (index, index) -> !fir.shapeshift<1>
  ! CHECK: %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_13]](%[[VAL_14]]) typeparams %[[VAL_2]] {uniq_name = "_QFdyn_array_cst_len_lbEc"} : (!fir.ref<!fir.array<?x!fir.char<1,10>>>, !fir.shapeshift<1>, index) -> (!fir.box<!fir.array<?x!fir.char<1,10>>>, !fir.ref<!fir.array<?x!fir.char<1,10>>>)
  print *, c(11)
end subroutine

! CHECK-LABEL: func @_QPdyn_array_dyn_len_lb
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<i64>{{.*}}, %[[ARG1:.*]]: !fir.ref<i64>
subroutine dyn_array_dyn_len_lb(l, n)
  integer(8) :: l, n
  character(l) :: c(11:n)
  ! CHECK: %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] arg 1 {uniq_name = "_QFdyn_array_dyn_len_lbEl"} : (!fir.ref<i64>, !fir.dscope) -> (!fir.ref<i64>, !fir.ref<i64>)
  ! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_0]] arg 2 {uniq_name = "_QFdyn_array_dyn_len_lbEn"} : (!fir.ref<i64>, !fir.dscope) -> (!fir.ref<i64>, !fir.ref<i64>)
  ! CHECK: %[[VAL_3:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<i64>
  ! CHECK: %[[VAL_4:.*]] = arith.constant 0 : i64
  ! CHECK: %[[VAL_5:.*]] = arith.cmpi sgt, %[[VAL_3]], %[[VAL_4]] : i64
  ! CHECK: %[[VAL_6:.*]] = arith.select %[[VAL_5]], %[[VAL_3]], %[[VAL_4]] : i64
  ! CHECK: %[[VAL_7:.*]] = arith.constant 11 : i64
  ! CHECK: %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i64) -> index
  ! CHECK: %[[VAL_9:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<i64>
  ! CHECK: %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i64) -> index
  ! CHECK: %[[VAL_11:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_12:.*]] = arith.constant 0 : index
  ! CHECK: %[[VAL_13:.*]] = arith.subi %[[VAL_10]], %[[VAL_8]] : index
  ! CHECK: %[[VAL_14:.*]] = arith.addi %[[VAL_13]], %[[VAL_11]] : index
  ! CHECK: %[[VAL_15:.*]] = arith.cmpi sgt, %[[VAL_14]], %[[VAL_12]] : index
  ! CHECK: %[[VAL_16:.*]] = arith.select %[[VAL_15]], %[[VAL_14]], %[[VAL_12]] : index
  ! CHECK: %[[VAL_17:.*]] = fir.alloca !fir.array<?x!fir.char<1,?>>(%[[VAL_6]] : i64), %[[VAL_16]] {bindc_name = "c", uniq_name = "_QFdyn_array_dyn_len_lbEc"}
  ! CHECK: %[[VAL_18:.*]] = fir.shape_shift %[[VAL_8]], %[[VAL_16]] : (index, index) -> !fir.shapeshift<1>
  ! CHECK: %[[VAL_19:.*]]:2 = hlfir.declare %[[VAL_17]](%[[VAL_18]]) typeparams %[[VAL_6]] {uniq_name = "_QFdyn_array_dyn_len_lbEc"} : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shapeshift<1>, i64) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.ref<!fir.array<?x!fir.char<1,?>>>)
  print *, c(11)
end subroutine

! Test that the length of assumed length parameter is correctly deduced in lowering.
! CHECK-LABEL: func @_QPassumed_length_param
subroutine assumed_length_param(n)
  character(*), parameter :: c(1)=(/"abcd"/)
  integer :: n
  ! CHECK: %[[VAL_0:.*]] = arith.constant 4 : i64
  ! CHECK: %[[VAL_1:.*]]:3 = hlfir.associate %[[VAL_0]] {adapt.valuebyref} : (i64) -> (!fir.ref<i64>, !fir.ref<i64>, i1)
  ! CHECK: fir.call @_QPtake_int(%[[VAL_1]]#0) {{.*}} : (!fir.ref<i64>) -> ()
  call take_int(len(c(n), kind=8))
end

! CHECK-LABEL: func @_QPscalar_cst_neg_len
subroutine scalar_cst_neg_len()
  character(-1) :: c
  ! CHECK: %[[VAL_0:.*]] = arith.constant 0 : index
  ! CHECK: %[[VAL_1:.*]] = fir.alloca !fir.char<1,0> {bindc_name = "c", uniq_name = "_QFscalar_cst_neg_lenEc"}
  ! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] typeparams %[[VAL_0]] {uniq_name = "_QFscalar_cst_neg_lenEc"} : (!fir.ref<!fir.char<1,0>>, index) -> (!fir.ref<!fir.char<1,0>>, !fir.ref<!fir.char<1,0>>)
  print *, c
end subroutine

! Test lowering of F77 calls to HLFIR
! RUN: bbc -emit-fir -hlfir -o - %s 2>&1 | FileCheck %s

! -----------------------------------------------------------------------------
!     Test lowering of F77 procedure reference arguments
! -----------------------------------------------------------------------------

subroutine call_no_arg()
  call void()
end subroutine
! CHECK-LABEL: func.func @_QPcall_no_arg() {
! CHECK-NEXT:  fir.call @_QPvoid() fastmath<contract> : () -> ()
! CHECK-NEXT:  return

subroutine call_int_arg_var(n)
  integer :: n
  call take_i4(n)
end subroutine
! CHECK-LABEL: func.func @_QPcall_int_arg_var(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<i32>
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFcall_int_arg_varEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:  fir.call @_QPtake_i4(%[[VAL_1]]#1) fastmath<contract> : (!fir.ref<i32>) -> ()

subroutine call_int_arg_expr()
  call take_i4(42)
end subroutine
! CHECK-LABEL: func.func @_QPcall_int_arg_expr() {
! CHECK:  %[[VAL_0:.*]] = arith.constant 42 : i32
! CHECK:  %[[VAL_1:.*]]:3 = hlfir.associate %[[VAL_0]] {uniq_name = "adapt.valuebyref"} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:  fir.call @_QPtake_i4(%[[VAL_1]]#1) fastmath<contract> : (!fir.ref<i32>) -> ()
! CHECK:  hlfir.end_associate %[[VAL_1]]#1, %[[VAL_1]]#2 : !fir.ref<i32>, i1

subroutine call_real_arg_expr()
  call take_r4(0.42)
end subroutine
! CHECK-LABEL: func.func @_QPcall_real_arg_expr() {
! CHECK:  %[[VAL_0:.*]] = arith.constant 4.200000e-01 : f32
! CHECK:  %[[VAL_1:.*]]:3 = hlfir.associate %[[VAL_0]] {uniq_name = "adapt.valuebyref"} : (f32) -> (!fir.ref<f32>, !fir.ref<f32>, i1)
! CHECK:  fir.call @_QPtake_r4(%[[VAL_1]]#1) fastmath<contract> : (!fir.ref<f32>) -> ()
! CHECK:  hlfir.end_associate %[[VAL_1]]#1, %[[VAL_1]]#2 : !fir.ref<f32>, i1

subroutine call_real_arg_var(x)
  real :: x
  call take_r4(x)
end subroutine
! CHECK-LABEL: func.func @_QPcall_real_arg_var(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<f32>
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFcall_real_arg_varEx"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:  fir.call @_QPtake_r4(%[[VAL_1]]#1) fastmath<contract> : (!fir.ref<f32>) -> ()

subroutine call_logical_arg_var(x)
  logical :: x
  call take_l4(x)
end subroutine
! CHECK-LABEL: func.func @_QPcall_logical_arg_var(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.logical<4>>
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFcall_logical_arg_varEx"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:  fir.call @_QPtake_l4(%[[VAL_1]]#1) fastmath<contract> : (!fir.ref<!fir.logical<4>>) -> ()

subroutine call_logical_arg_expr()
  call take_l4(.true.)
end subroutine
! CHECK-LABEL: func.func @_QPcall_logical_arg_expr() {
! CHECK:  %[[VAL_0:.*]] = arith.constant true
! CHECK:  %[[VAL_1:.*]] = fir.convert %[[VAL_0]] : (i1) -> !fir.logical<4>
! CHECK:  %[[VAL_2:.*]]:3 = hlfir.associate %[[VAL_1]] {uniq_name = "adapt.valuebyref"} : (!fir.logical<4>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>, i1)
! CHECK:  fir.call @_QPtake_l4(%[[VAL_2]]#1) fastmath<contract> : (!fir.ref<!fir.logical<4>>) -> ()
! CHECK:  hlfir.end_associate %[[VAL_2]]#1, %[[VAL_2]]#2 : !fir.ref<!fir.logical<4>>, i1

subroutine call_logical_arg_expr_2()
  call take_l8(.true._8)
end subroutine
! CHECK-LABEL: func.func @_QPcall_logical_arg_expr_2() {
! CHECK:  %[[VAL_0:.*]] = arith.constant true
! CHECK:  %[[VAL_1:.*]] = fir.convert %[[VAL_0]] : (i1) -> !fir.logical<8>
! CHECK:  %[[VAL_2:.*]]:3 = hlfir.associate %[[VAL_1]] {uniq_name = "adapt.valuebyref"} : (!fir.logical<8>) -> (!fir.ref<!fir.logical<8>>, !fir.ref<!fir.logical<8>>, i1)
! CHECK:  fir.call @_QPtake_l8(%[[VAL_2]]#1) fastmath<contract> : (!fir.ref<!fir.logical<8>>) -> ()
! CHECK:  hlfir.end_associate %[[VAL_2]]#1, %[[VAL_2]]#2 : !fir.ref<!fir.logical<8>>, i1

subroutine call_char_arg_var(x)
  character(*) :: x
  call take_c(x)
end subroutine
! CHECK-LABEL: func.func @_QPcall_char_arg_var(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.boxchar<1>
! CHECK:  %[[VAL_1:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]]#0 typeparams %[[VAL_1]]#1 {uniq_name = "_QFcall_char_arg_varEx"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:  fir.call @_QPtake_c(%[[VAL_2]]#0) fastmath<contract> : (!fir.boxchar<1>) -> ()

subroutine call_char_arg_var_expr(x)
  character(*) :: x
  call take_c(x//x)
end subroutine
! CHECK-LABEL: func.func @_QPcall_char_arg_var_expr(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.boxchar<1>
! CHECK:  %[[VAL_1:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]]#0 typeparams %[[VAL_1]]#1 {uniq_name = "_QFcall_char_arg_var_exprEx"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:  %[[VAL_3:.*]] = arith.addi %[[VAL_1]]#1, %[[VAL_1]]#1 : index
! CHECK:  %[[VAL_4:.*]] = hlfir.concat %[[VAL_2]]#0, %[[VAL_2]]#0 len %[[VAL_3]] : (!fir.boxchar<1>, !fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,?>>
! CHECK:  %[[VAL_5:.*]]:3 = hlfir.associate %[[VAL_4]] typeparams %[[VAL_3]] {uniq_name = "adapt.valuebyref"} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
! CHECK:  fir.call @_QPtake_c(%[[VAL_5]]#0) fastmath<contract> : (!fir.boxchar<1>) -> ()
! CHECK:  hlfir.end_associate %[[VAL_5]]#1, %[[VAL_5]]#2 : !fir.ref<!fir.char<1,?>>, i1

subroutine call_arg_array_var(n)
  integer :: n(10, 20)
  call take_arr(n)
end subroutine
! CHECK-LABEL: func.func @_QPcall_arg_array_var(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<10x20xi32>>
! CHECK:  %[[VAL_1:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_2:.*]] = arith.constant 20 : index
! CHECK:  %[[VAL_3:.*]] = fir.shape %[[VAL_1]], %[[VAL_2]] : (index, index) -> !fir.shape<2>
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_3]]) {uniq_name = "_QFcall_arg_array_varEn"} : (!fir.ref<!fir.array<10x20xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<10x20xi32>>, !fir.ref<!fir.array<10x20xi32>>)
! CHECK:  fir.call @_QPtake_arr(%[[VAL_4]]#1) fastmath<contract> : (!fir.ref<!fir.array<10x20xi32>>) -> ()

subroutine call_arg_array_2(n)
  integer, contiguous, optional :: n(:, :)
  call take_arr_2(n)
end subroutine
! CHECK-LABEL: func.func @_QPcall_arg_array_2(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xi32>>
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<contiguous, optional>, uniq_name = "_QFcall_arg_array_2En"} : (!fir.box<!fir.array<?x?xi32>>) -> (!fir.box<!fir.array<?x?xi32>>, !fir.box<!fir.array<?x?xi32>>)
! CHECK:  %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]]#1 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.ref<!fir.array<?x?xi32>>
! CHECK:  fir.call @_QPtake_arr_2(%[[VAL_2]]) fastmath<contract> : (!fir.ref<!fir.array<?x?xi32>>) -> ()

! -----------------------------------------------------------------------------
!     Test lowering of function results
! -----------------------------------------------------------------------------

subroutine return_integer()
  integer :: ifoo
  print *, ifoo()
end subroutine
! CHECK-LABEL: func.func @_QPreturn_integer(
! CHECK:  fir.call @_QPifoo() fastmath<contract> : () -> i32


subroutine return_logical()
  logical :: lfoo
  print *, lfoo()
end subroutine
! CHECK-LABEL: func.func @_QPreturn_logical(
! CHECK:  fir.call @_QPlfoo() fastmath<contract> : () -> !fir.logical<4>

subroutine return_complex()
  complex :: cplxfoo
  print *, cplxfoo()
end subroutine
! CHECK-LABEL: func.func @_QPreturn_complex(
! CHECK:  fir.call @_QPcplxfoo() fastmath<contract> : () -> !fir.complex<4>

subroutine return_char(n)
  integer(8) :: n
  character(n) :: c2foo
  print *, c2foo()
end subroutine
! CHECK-LABEL: func.func @_QPreturn_char(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare {{.*}}n
! CHECK:  %[[VAL_2:.*]] = arith.constant -1 : i32
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_1]]#1 : !fir.ref<i64>
! CHECK:  %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i64) -> index
! CHECK:  %[[VAL_9:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_10:.*]] = arith.cmpi sgt, %[[VAL_8]], %[[VAL_9]] : index
! CHECK:  %[[VAL_11:.*]] = arith.select %[[VAL_10]], %[[VAL_8]], %[[VAL_9]] : index
! CHECK:  %[[VAL_13:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_11]] : index) {bindc_name = ".result"}
! CHECK:  %[[VAL_14:.*]] = fir.call @_QPc2foo(%[[VAL_13]], %[[VAL_11]]) fastmath<contract> : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:  %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_13]] typeparams %[[VAL_11]] {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)

! -----------------------------------------------------------------------------
!     Test calls with alternate returns
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPalternate_return_call(
subroutine alternate_return_call(n1, n2, k)
  integer :: n1, n2, k
  ! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare {{.*}}k
  ! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare {{.*}}n1
  ! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare {{.*}}n2
  ! CHECK:  %[[selector:.*]] = fir.call @_QPalternate_return(%[[VAL_4]]#1, %[[VAL_5]]#1) fastmath<contract> : (!fir.ref<i32>, !fir.ref<i32>) -> index
  ! CHECK-NEXT: fir.select %[[selector]] : index [1, ^[[block1:bb[0-9]+]], 2, ^[[block2:bb[0-9]+]], unit, ^[[blockunit:bb[0-9]+]]
  call alternate_return(n1, *5, n2, *7)
  ! CHECK: ^[[blockunit]]: // pred: ^bb0
  k =  0; return;
  ! CHECK: ^[[block1]]: // pred: ^bb0
5 k = -1; return;
  ! CHECK: ^[[block2]]: // pred: ^bb0
7 k =  1; return
end

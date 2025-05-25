! Test lowering of statement functions to HLFIR
! RUN: bbc -emit-hlfir -o - %s 2>&1 | FileCheck %s

subroutine numeric_test(x)
  integer :: x(:), i, stmt_func
  stmt_func(i) = x(i)
  call bar(stmt_func(42))
end subroutine
! CHECK-LABEL: func.func @_QPnumeric_test(
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0:[^)]*]] {{.*}}x"
! CHECK:  %[[VAL_6:.*]] = arith.constant 42 : i32
! CHECK:  %[[VAL_7:.*]]:3 = hlfir.associate %[[VAL_6]] {uniq_name = "i"} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:  %[[VAL_8:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<i32>
! CHECK:  %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i32) -> i64
! CHECK:  %[[VAL_10:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_9]])  : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
! CHECK:  %[[VAL_11:.*]] = fir.load %[[VAL_10]] : !fir.ref<i32>

subroutine char_test(c, n)
  character(*) :: c
  character(n) :: char_stmt_func_dummy_arg
  character(10) :: stmt_func
  stmt_func(char_stmt_func_dummy_arg) = char_stmt_func_dummy_arg
  call bar2(stmt_func(c))
end subroutine
! CHECK-LABEL: func.func @_QPchar_test(
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3:.*]]#0 typeparams %[[VAL_3]]#1 {{.*}}c"
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_2:[^ ]*]] {{.*}}n"
! CHECK:  %[[VAL_13:.*]]:2 = fir.unboxchar %[[VAL_4]]#0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_14:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
! CHECK:  %[[VAL_15:.*]] = arith.constant 0 : i32
! CHECK:  %[[VAL_16:.*]] = arith.cmpi sgt, %[[VAL_14]], %[[VAL_15]] : i32
! CHECK:  %[[VAL_17:.*]] = arith.select %[[VAL_16]], %[[VAL_14]], %[[VAL_15]] : i32
! CHECK:  %[[VAL_18:.*]]:2 = hlfir.declare %[[VAL_13]]#0 typeparams %[[VAL_17]] {uniq_name = "_QFchar_testFstmt_funcEchar_stmt_func_dummy_arg"} : (!fir.ref<!fir.char<1,?>>, i32) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:  %[[VAL_19:.*]] = arith.constant 10 : i64
! CHECK:  %[[VAL_20:.*]] = hlfir.set_length %[[VAL_18]]#0 len %[[VAL_19]] : (!fir.boxchar<1>, i64) -> !hlfir.expr<!fir.char<1,10>>

subroutine char_test2(c)
  character(10) :: c
  character(5) :: c_stmt_func
  character(*), parameter :: padding = "padding"
  character(len(c_stmt_func)+len(padding)) :: stmt_func
  stmt_func(c_stmt_func) = c_stmt_func // padding
  call test(stmt_func(c))
end subroutine
! CHECK-LABEL:  func.func @_QPchar_test2(
! CHECK:    %[[C:.*]]:2 = hlfir.declare %{{.*}} typeparams %c10 dummy_scope %{{[0-9]+}} {uniq_name = "_QFchar_test2Ec"} : (!fir.ref<!fir.char<1,10>>, index, !fir.dscope) -> (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>)
! CHECK:    %[[CAST:.*]] = fir.convert %[[C]]#0 : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<!fir.char<1,5>>
! CHECK:    %[[C_STMT_FUNC:.*]]:2 = hlfir.declare %[[CAST]] typeparams %c5{{.*}} {uniq_name = "_QFchar_test2Fstmt_funcEc_stmt_func"} : (!fir.ref<!fir.char<1,5>>, index) -> (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>)
! CHECK:    hlfir.concat %[[C_STMT_FUNC]]#0, %{{.*}} len %{{.*}} : (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,7>>, index) -> !hlfir.expr<!fir.char<1,12>>

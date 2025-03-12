! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

subroutine test_chdir()
  implicit none
! CHECK-LABEL:   func.func @_QPtest_chdir() {

  call chdir("..")
! CHECK:  %[[VAL_0:.*]] = fir.address_of(@_QQclX2E2E) : !fir.ref<!fir.char<1,2>>
! CHECK:  %[[C_2:.*]] = arith.constant 2 : index
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[C_2]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX2E2E"} : (!fir.ref<!fir.char<1,2>>, index) -> (!fir.ref<!fir.char<1,2>>, !fir.ref<!fir.char<1,2>>)
! CHECK:  %[[VAL_2:.*]] = fir.convert %[[VAL_1]]#1 : (!fir.ref<!fir.char<1,2>>) -> !fir.ref<i8>
! CHECK:  %[[VAL_3:.*]] = fir.call @_FortranAChdir(%[[VAL_2]]) fastmath<contract> : (!fir.ref<i8>) -> i32
end subroutine

subroutine test_chdir_subroutine_status_i4()
  implicit none
  integer(4) :: stat
! CHECK-LABEL:   func.func @_QPtest_chdir_subroutine_status_i4() {

  call chdir("..", STATUS=stat)
! CHECK:  %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "stat", uniq_name = "_QFtest_chdir_subroutine_status_i4Estat"}
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtest_chdir_subroutine_status_i4Estat"} : (!fir.ref<i32>) ->
! (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:  %[[VAL_2:.*]] = fir.address_of(@_QQclX2E2E) : !fir.ref<!fir.char<1,2>>
! CHECK:  %[[C_2:.*]] = arith.constant 2 : index
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] typeparams %[[C_2]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = {{.*}} : (!fir.ref<!fir.char<1,2>>, index) -> (!fir.ref<!fir.char<1,2>>, !fir.ref<!fir.char<1,2>>)
! CHECK:  %[[VAL_4:.*]] = fir.convert %[[VAL_3]]#1 : (!fir.ref<!fir.char<1,2>>) -> !fir.ref<i8>
! CHECK:  %[[VAL_5:.*]] = fir.call @_FortranAChdir(%[[VAL_4]]) fastmath<contract> : (!fir.ref<i8>) -> i32
! CHECK:  %[[VAL_6:.*]] = fir.convert %{{.*}} : (!fir.ref<i32>) -> i64
! CHECK:  %[[C_0_I64:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_7:.*]] = arith.cmpi ne, %[[VAL_6]], %[[C_0_I64]] : i64
! CHECK: fir.if %[[VAL_7]] {
! CHECK:   fir.store %[[VAL_5]] to %[[VAL_1]]#1 : !fir.ref<i32>
! CHECK: }
end subroutine

subroutine test_chdir_function_status_i4()
  implicit none
  integer(4) :: stat
! CHECK-LABEL:   func.func @_QPtest_chdir_function_status_i4() {

  stat = chdir("..")
! CHECK:  %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "stat", uniq_name = "_QFtest_chdir_function_status_i4Estat"}
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtest_chdir_function_status_i4Estat"} : (!fir.ref<i32>) ->
! (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:  %[[VAL_2:.*]] = fir.address_of(@_QQclX2E2E) : !fir.ref<!fir.char<1,2>>
! CHECK:  %[[C_2:.*]] = arith.constant 2 : index
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] typeparams %[[C_2]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = {{.*}} : (!fir.ref<!fir.char<1,2>>, index) -> (!fir.ref<!fir.char<1,2>>, !fir.ref<!fir.char<1,2>>)
! CHECK:  %[[VAL_4:.*]] = fir.convert %[[VAL_3]]#1 : (!fir.ref<!fir.char<1,2>>) -> !fir.ref<i8>
! CHECK:  %[[VAL_5:.*]] = fir.call @_FortranAChdir(%[[VAL_4]]) fastmath<contract> : (!fir.ref<i8>) -> i32
! CHECK: hlfir.assign %[[VAL_5]] to %[[VAL_1]]#0 : i32, !fir.ref<i32>
end subroutine

subroutine test_chdir_subroutine_status_i8()
  implicit none
  integer(8) :: stat
! CHECK-LABEL:   func.func @_QPtest_chdir_subroutine_status_i8() {

  call chdir("..", STATUS=stat)
! CHECK:  %[[VAL_0:.*]] = fir.alloca i64 {bindc_name = "stat", uniq_name = "_QFtest_chdir_subroutine_status_i8Estat"}
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtest_chdir_subroutine_status_i8Estat"} : (!fir.ref<i64>) ->
! (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:  %[[VAL_2:.*]] = fir.address_of(@_QQclX2E2E) : !fir.ref<!fir.char<1,2>>
! CHECK:  %[[C_2:.*]] = arith.constant 2 : index
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] typeparams %[[C_2]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = {{.*}} : (!fir.ref<!fir.char<1,2>>, index) -> (!fir.ref<!fir.char<1,2>>, !fir.ref<!fir.char<1,2>>)
! CHECK:  %[[VAL_4:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.char<1,2>>) -> !fir.ref<i8>
! CHECK:  %[[VAL_5:.*]] = fir.call @_FortranAChdir(%[[VAL_4]]) fastmath<contract> : (!fir.ref<i8>) -> i32
! CHECK:  %[[VAL_6:.*]] = fir.convert %[[VAL_1]]#1 : (!fir.ref<i64>) -> i64
! CHECK:  %[[C_0_I64:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_7:.*]] = arith.cmpi ne, %[[VAL_6]], %[[C_0_I64]] : i64
! CHECK: fir.if %[[VAL_7]] {
! CHECK:  %[[VAL_8:.*]] = fir.convert %[[VAL_5]] : (i32) -> i64
! CHECK:   fir.store %[[VAL_8]] to %[[VAL_1]]#1 : !fir.ref<i64>
! CHECK: }
end subroutine

subroutine test_chdir_function_status_i8()
  implicit none
  integer(8) :: stat
! CHECK-LABEL:   func.func @_QPtest_chdir_function_status_i8() {

  stat = chdir("..")
! CHECK:  %[[VAL_0:.*]] = fir.alloca i64 {bindc_name = "stat", uniq_name = "_QFtest_chdir_function_status_i8Estat"}
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtest_chdir_function_status_i8Estat"} : (!fir.ref<i64>) ->
! (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:  %[[VAL_2:.*]] = fir.address_of(@_QQclX2E2E) : !fir.ref<!fir.char<1,2>>
! CHECK:  %[[C_2:.*]] = arith.constant 2 : index
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] typeparams %[[C_2]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = {{.*}} : (!fir.ref<!fir.char<1,2>>, index) -> (!fir.ref<!fir.char<1,2>>, !fir.ref<!fir.char<1,2>>)
! CHECK:  %[[VAL_4:.*]] = fir.convert %[[VAL_3]]#1 : (!fir.ref<!fir.char<1,2>>) -> !fir.ref<i8>
! CHECK:  %[[VAL_5:.*]] = fir.call @_FortranAChdir(%[[VAL_4]]) fastmath<contract> : (!fir.ref<i8>) -> i32
! CHECK:  %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i32) -> i64
! CHECK: hlfir.assign %[[VAL_6]] to %[[VAL_1]]#0 : i64, !fir.ref<i64>
end subroutine


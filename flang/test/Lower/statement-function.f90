! RUN: %flang_fc1 -emit-hlfir -outline-intrinsics %s -o - | FileCheck %s

! Test statement function lowering

! Simple case
  ! CHECK-LABEL: func @_QPtest_stmt_0(
  ! CHECK-SAME: %{{.*}}: !fir.ref<f32>{{.*}}) -> f32
real function test_stmt_0(x)
  real :: x, func, arg
  func(arg) = arg + 0.123456

  ! CHECK: %[[res:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_stmt_0Etest_stmt_0"}
  ! CHECK: %[[xdecl:.*]]:2 = hlfir.declare %arg0 {{.*}} {uniq_name = "_QFtest_stmt_0Ex"}
  ! CHECK: %[[x:.*]] = fir.load %[[xdecl]]#0
  ! CHECK: %[[cst:.*]] = arith.constant 1.234560e-01
  ! CHECK: %[[eval:.*]] = arith.addf %[[x]], %[[cst]]
  ! CHECK: hlfir.assign %[[eval]] to %[[res]]#0 : f32, !fir.ref<f32>
  test_stmt_0 = func(x)

  ! CHECK: %[[resval:.*]] = fir.load %[[res]]#0
  ! CHECK: return %[[resval]]
end function

! Check this is not lowered as a simple macro: e.g. argument is only
! evaluated once even if it appears in several placed inside the
! statement function expression
! CHECK-LABEL: func @_QPtest_stmt_only_eval_arg_once() -> f32
real(4) function test_stmt_only_eval_arg_once()
  real(4) :: only_once, x1
  func(x1) = x1 + x1
  ! CHECK: %[[call:.*]] = fir.call @_QPonly_once()
  ! CHECK: %[[assoc:.*]]:3 = hlfir.associate %[[call]]
  ! CHECK: %[[v1:.*]] = fir.load %[[assoc]]#0
  ! CHECK: %[[v2:.*]] = fir.load %[[assoc]]#0
  ! CHECK: arith.addf %[[v1]], %[[v2]]
  test_stmt_only_eval_arg_once = func(only_once())
end function

! Test nested statement function (note that they cannot be recursively
! nested as per F2018 C1577).
real function test_stmt_1(x, a)
  real :: y, a, b, foo
  real :: func1, arg1, func2, arg2
  real :: res1, res2
  func1(arg1) = a + foo(arg1)
  func2(arg2) = func1(arg2) + b
  ! CHECK-DAG: %[[adecl:.*]]:2 = hlfir.declare %arg1 {{.*}} {uniq_name = "_QFtest_stmt_1Ea"}
  ! CHECK-DAG: %[[bmem:.*]] = fir.alloca f32 {{{.*}}uniq_name = "_QFtest_stmt_1Eb"}
  ! CHECK-DAG: %[[bdecl:.*]]:2 = hlfir.declare %[[bmem]] {uniq_name = "_QFtest_stmt_1Eb"}
  ! CHECK-DAG: %[[res1:.*]] = fir.alloca f32 {{{.*}}uniq_name = "_QFtest_stmt_1Eres1"}
  ! CHECK-DAG: %[[res1decl:.*]]:2 = hlfir.declare %[[res1]] {uniq_name = "_QFtest_stmt_1Eres1"}
  ! CHECK-DAG: %[[res2:.*]] = fir.alloca f32 {{{.*}}uniq_name = "_QFtest_stmt_1Eres2"}
  ! CHECK-DAG: %[[res2decl:.*]]:2 = hlfir.declare %[[res2]] {uniq_name = "_QFtest_stmt_1Eres2"}
  ! CHECK-DAG: %[[xdecl:.*]]:2 = hlfir.declare %arg0 {{.*}} {uniq_name = "_QFtest_stmt_1Ex"}

  b = 5

  ! CHECK-DAG: %[[cst_8:.*]] = arith.constant 8.000000e+00
  ! CHECK-DAG: %[[assoc1:.*]]:3 = hlfir.associate %[[cst_8]]
  ! CHECK-DAG: %[[aload1:.*]] = fir.load %[[adecl]]#0
  ! CHECK-DAG: %[[foocall1:.*]] = fir.call @_QPfoo(%[[assoc1]]#0)
  ! CHECK: %[[add1:.*]] = arith.addf %[[aload1]], %[[foocall1]]
  ! CHECK: hlfir.assign %[[add1]] to %[[res1decl]]#0
  res1 =  func1(8.)

  ! CHECK-DAG: %[[a2:.*]] = fir.load %[[adecl]]#0
  ! CHECK-DAG: %[[foocall2:.*]] = fir.call @_QPfoo(%[[xdecl]]#0)
  ! CHECK-DAG: %[[add2:.*]] = arith.addf %[[a2]], %[[foocall2]]
  ! CHECK-DAG: %[[b:.*]] = fir.load %[[bdecl]]#0
  ! CHECK: %[[add3:.*]] = arith.addf %[[add2]], %[[b]]
  ! CHECK: hlfir.assign %[[add3]] to %[[res2decl]]#0
  res2 = func2(x)

  ! CHECK-DAG: %[[res12:.*]] = fir.load %[[res1decl]]#0
  ! CHECK-DAG: %[[res22:.*]] = fir.load %[[res2decl]]#0
  ! CHECK: = arith.addf %[[res12]], %[[res22]] {{.*}}: f32
  test_stmt_1 = res1 + res2
  ! CHECK: return %{{.*}} : f32
end function


! Test statement functions with no argument.
! Test that they are not pre-evaluated.
! CHECK-LABEL: func @_QPtest_stmt_no_args
real function test_stmt_no_args(x, y)
  func() = x + y
  ! CHECK: addf
  a = func()
  ! CHECK: fir.call @_QPfoo_may_modify_xy
  call foo_may_modify_xy(x, y)
  ! CHECK: addf
  ! CHECK: addf
  test_stmt_no_args = func() + a
end function

! Test statement function with character arguments
! CHECK-LABEL: @_QPtest_stmt_character
integer function test_stmt_character(c, j)
  integer :: i, j, func, argj
  character(10) :: c, argc
  ! CHECK: %[[unboxed:.*]]:2 = fir.unboxchar %arg0 :
  ! CHECK: %[[ref:.*]] = fir.convert %[[unboxed]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,10>>
  ! CHECK: %[[cdecl:.*]]:2 = hlfir.declare %[[ref]] typeparams %{{.*}} {{.*}} {uniq_name = "_QFtest_stmt_characterEc"}
  ! CHECK: %[[funcArgcDecl:.*]]:2 = hlfir.declare %[[cdecl]]#0 typeparams %{{.*}} {uniq_name = "_QFtest_stmt_characterFfuncEargc"}

  func(argc, argj) = len_trim(argc, 4) + argj
  ! CHECK: addi %{{.*}}, %{{.*}} : i
  test_stmt_character = func(c, j)
end function


! Test statement function with a character actual argument whose
! length may be different than the dummy length (the dummy length
! must be used inside the statement function).
! CHECK-LABEL: @_QPtest_stmt_character_with_different_length(
! CHECK-SAME: %[[arg0:.*]]: !fir.boxchar<1>
integer function test_stmt_character_with_different_length(c)
  integer :: func, ifoo
  character(10) :: argc
  character(*) :: c
  ! CHECK: %[[unboxedC:.*]]:2 = fir.unboxchar %[[arg0]] :
  ! CHECK: %[[cdecl:.*]]:2 = hlfir.declare %[[unboxedC]]#0 typeparams %[[unboxedC]]#1 {{.*}} {uniq_name = "_QFtest_stmt_character_with_different_lengthEc"}
  ! CHECK: %[[unboxedArg:.*]]:2 = fir.unboxchar %[[cdecl]]#0 :
  ! CHECK: %[[ref:.*]] = fir.convert %[[unboxedArg]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,10>>
  ! CHECK: %[[c10:.*]] = arith.constant 10 : index
  ! CHECK: %[[funcArgc:.*]]:2 = hlfir.declare %[[ref]] typeparams %[[c10]] {uniq_name = "_QFtest_stmt_character_with_different_lengthFfuncEargc"}
  ! CHECK: %[[argc:.*]] = fir.emboxchar %[[funcArgc]]#0, %[[c10]]
  ! CHECK: fir.call @_QPifoo(%[[argc]]) {{.*}}: (!fir.boxchar<1>) -> i32
  func(argc) = ifoo(argc)
  test_stmt_character = func(c)
end function

! CHECK-LABEL: @_QPtest_stmt_character_with_different_length_2(
! CHECK-SAME: %[[arg0:.*]]: !fir.boxchar<1>{{.*}}, %[[arg1:.*]]: !fir.ref<i32>
integer function test_stmt_character_with_different_length_2(c, n)
  integer :: func, ifoo
  character(n) :: argc
  character(*) :: c
  ! CHECK: %[[ndecl:.*]]:2 = hlfir.declare %[[arg1]] {{.*}} {uniq_name = "_QFtest_stmt_character_with_different_length_2En"}
  ! CHECK: %[[unboxedC:.*]]:2 = fir.unboxchar %[[arg0]] :
  ! CHECK: %[[cdecl:.*]]:2 = hlfir.declare %[[unboxedC]]#0 typeparams %[[unboxedC]]#1 {{.*}} {uniq_name = "_QFtest_stmt_character_with_different_length_2Ec"}
  ! CHECK: %[[unboxedArg:.*]]:2 = fir.unboxchar %[[cdecl]]#0 :
  ! CHECK: %[[n:.*]] = fir.load %[[ndecl]]#0 : !fir.ref<i32>
  ! CHECK: %[[n_is_positive:.*]] = arith.cmpi sgt, %[[n]], %c0{{.*}} : i32
  ! CHECK: %[[len:.*]] = arith.select %[[n_is_positive]], %[[n]], %c0{{.*}} : i32
  ! CHECK: %[[funcArgc:.*]]:2 = hlfir.declare %[[unboxedArg]]#0 typeparams %[[len]] {uniq_name = "_QFtest_stmt_character_with_different_length_2FfuncEargc"} : (!fir.ref<!fir.char<1,?>>, i32) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  ! CHECK: fir.call @_QPifoo(%[[funcArgc]]#0) {{.*}}: (!fir.boxchar<1>) -> i32
  func(argc) = ifoo(argc)
  test_stmt_character = func(c)
end function

! issue #247
! CHECK-LABEL: @_QPbug247
subroutine bug247(r)
  I(R) = R
  ! CHECK: fir.call {{.*}}OutputInteger
  PRINT *, I(2.5)
  ! CHECK: fir.call {{.*}}EndIo
END subroutine bug247

! Test that the argument is truncated to the length of the dummy argument.
subroutine truncate_arg
  character(4) arg
  character(10) stmt_fct
  stmt_fct(arg) = arg
  print *, stmt_fct('longer_arg')
end subroutine

! CHECK-LABEL: @_QPtruncate_arg
! CHECK: %[[arg:.*]] = fir.address_of(@_QQclX{{.*}}) : !fir.ref<!fir.char<1,10>>
! CHECK: %[[c10:.*]] = arith.constant 10 : index
! CHECK: %[[argDecl:.*]]:2 = hlfir.declare %[[arg]] typeparams %[[c10]]
! CHECK: %[[castArg:.*]] = fir.convert %[[argDecl]]#0 : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<!fir.char<1,4>>
! CHECK: %[[c4:.*]] = arith.constant 4 : index
! CHECK: %[[fctArg:.*]]:2 = hlfir.declare %[[castArg]] typeparams %[[c4]] {uniq_name = "_QFtruncate_argFstmt_fctEarg"}
! CHECK: %[[c10_i64:.*]] = arith.constant 10 : i64
! CHECK: %[[setlen:.*]] = hlfir.set_length %[[fctArg]]#0 len %[[c10_i64]] : (!fir.ref<!fir.char<1,4>>, i64) -> !hlfir.expr<!fir.char<1,10>>
! CHECK: %[[assoc:.*]]:3 = hlfir.associate %[[setlen]] typeparams %[[c10_i64]] {{.*}} : (!hlfir.expr<!fir.char<1,10>>, i64) -> (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>, i1)
! CHECK: %[[castTemp:.*]] = fir.convert %[[assoc]]#0 : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
! CHECK: %{{.*}} = fir.call @_FortranAioOutputAscii(%{{.*}}, %[[castTemp]], %[[c10_i64]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1

! RUN: bbc -emit-fir -outline-intrinsics %s -o - | FileCheck %s

! Test statement function lowering

! Simple case
  ! CHECK-LABEL: func @_QPtest_stmt_0(
  ! CHECK-SAME: %{{.*}}: !fir.ref<f32>{{.*}}) -> f32
real function test_stmt_0(x)
  real :: x, func, arg
  func(arg) = arg + 0.123456

  ! CHECK-DAG: %[[x:.*]] = fir.load %arg0
  ! CHECK-DAG: %[[cst:.*]] = arith.constant 1.234560e-01
  ! CHECK: %[[eval:.*]] = arith.addf %[[x]], %[[cst]]
  ! CHECK: fir.store %[[eval]] to %[[resmem:.*]] : !fir.ref<f32>
  test_stmt_0 = func(x)

  ! CHECK: %[[res:.*]] = fir.load %[[resmem]]
  ! CHECK: return %[[res]]
end function

! Check this is not lowered as a simple macro: e.g. argument is only
! evaluated once even if it appears in several placed inside the
! statement function expression 
! CHECK-LABEL: func @_QPtest_stmt_only_eval_arg_once() -> f32
real(4) function test_stmt_only_eval_arg_once()
  real(4) :: only_once, x1
  func(x1) = x1 + x1
  ! CHECK: %[[x2:.*]] = fir.alloca f32 {adapt.valuebyref}
  ! CHECK: %[[x1:.*]] = fir.call @_QPonly_once()
  ! Note: using -emit-fir, so the faked pass-by-reference is exposed
  ! CHECK: fir.store %[[x1]] to %[[x2]]
  ! CHECK: addf %{{.*}}, %{{.*}}
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
  ! CHECK-DAG: %[[bmem:.*]] = fir.alloca f32 {{{.*}}uniq_name = "{{.*}}Eb"}
  ! CHECK-DAG: %[[res1:.*]] = fir.alloca f32 {{{.*}}uniq_name = "{{.*}}Eres1"}
  ! CHECK-DAG: %[[res2:.*]] = fir.alloca f32 {{{.*}}uniq_name = "{{.*}}Eres2"}

  b = 5

  ! CHECK-DAG: %[[cst_8:.*]] = arith.constant 8.000000e+00
  ! CHECK-DAG: fir.store %[[cst_8]] to %[[tmp1:.*]] : !fir.ref<f32>
  ! CHECK-DAG: %[[foocall1:.*]] = fir.call @_QPfoo(%[[tmp1]])
  ! CHECK-DAG: %[[aload1:.*]] = fir.load %arg1
  ! CHECK: %[[add1:.*]] = arith.addf %[[aload1]], %[[foocall1]]
  ! CHECK: fir.store %[[add1]] to %[[res1]]
  res1 =  func1(8.)

  ! CHECK-DAG: %[[a2:.*]] = fir.load %arg1
  ! CHECK-DAG: %[[foocall2:.*]] = fir.call @_QPfoo(%arg0)
  ! CHECK-DAG: %[[add2:.*]] = arith.addf %[[a2]], %[[foocall2]]
  ! CHECK-DAG: %[[b:.*]] = fir.load %[[bmem]]
  ! CHECK: %[[add3:.*]] = arith.addf %[[add2]], %[[b]]
  ! CHECK: fir.store %[[add3]] to %[[res2]]
  res2 = func2(x)

  ! CHECK-DAG: %[[res12:.*]] = fir.load %[[res1]]
  ! CHECK-DAG: %[[res22:.*]] = fir.load %[[res2]]
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
  ! CHECK-DAG: %[[unboxed:.*]]:2 = fir.unboxchar %arg0 :
  ! CHECK-DAG: %[[c10:.*]] = arith.constant 10 :
  ! CHECK: %[[c10_cast:.*]] = fir.convert %[[c10]] : (i32) -> index
  ! CHECK: %[[c:.*]] = fir.emboxchar %[[unboxed]]#0, %[[c10_cast]]

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
  ! CHECK-DAG: %[[unboxed:.*]]:2 = fir.unboxchar %[[arg0]] :
  ! CHECK-DAG: %[[c10:.*]] = arith.constant 10 :
  ! CHECK: %[[c10_cast:.*]] = fir.convert %[[c10]] : (i32) -> index
  ! CHECK: %[[argc:.*]] = fir.emboxchar %[[unboxed]]#0, %[[c10_cast]]
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
  ! CHECK: %[[unboxed:.*]]:2 = fir.unboxchar %[[arg0]] :
  ! CHECK: fir.load %[[arg1]] : !fir.ref<i32>
  ! CHECK: %[[n:.*]] = fir.load %[[arg1]] : !fir.ref<i32>
  ! CHECK: %[[n_is_positive:.*]] = arith.cmpi sgt, %[[n]], %c0{{.*}} : i32
  ! CHECK: %[[len:.*]] = arith.select %[[n_is_positive]], %[[n]], %c0{{.*}} : i32
  ! CHECK: %[[lenCast:.*]] = fir.convert %[[len]] : (i32) -> index
  ! CHECK: %[[argc:.*]] = fir.emboxchar %[[unboxed]]#0, %[[lenCast]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK: fir.call @_QPifoo(%[[argc]]) {{.*}}: (!fir.boxchar<1>) -> i32
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
! CHECK: %[[c4:.*]] = arith.constant 4 : i32
! CHECK: %[[arg:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,10>>
! CHECK: %[[cast_arg:.*]] = fir.convert %[[arg]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<!fir.char<1,?>>
! CHECK: %[[c10:.*]] = arith.constant 10 : i64
! CHECK: %[[temp:.*]] = fir.alloca !fir.char<1,10> {bindc_name = ".chrtmp"}
! CHECK: %[[c10_index:.*]] = fir.convert %[[c10]] : (i64) -> index
! CHECK: %[[c4_index:.*]] = fir.convert %[[c4]] : (i32) -> index
! CHECK: %[[cmpi:.*]] = arith.cmpi slt, %[[c10_index]], %[[c4_index]] : index
! CHECK: %[[select:.*]] = arith.select %[[cmpi]], %[[c10_index]], %[[c4_index]] : index
! CHECK: %[[c1:.*]] = arith.constant 1 : i64
! CHECK: %[[select_i64:.*]] = fir.convert %[[select]] : (index) -> i64
! CHECK: %[[length:.*]] = arith.muli %[[c1]], %[[select_i64]] : i64
! CHECK: %[[cast_temp_i8:.*]] = fir.convert %[[temp]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
! CHECK: %[[cast_arg_i8:.*]] = fir.convert %[[cast_arg]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK: fir.call @llvm.memmove.p0.p0.i64(%[[cast_temp_i8]], %[[cast_arg_i8]], %[[length]], %{{.*}}) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! CHECK: %[[c1_i64:.*]] = arith.constant 1 : i64
! CHECK: %[[ub:.*]] = arith.subi %[[c10]], %[[c1_i64]] : i64
! CHECK: %[[ub_index:.*]] = fir.convert %[[ub]] : (i64) -> index
! CHECK: fir.do_loop %{{.*}} = %[[select]] to %[[ub_index]] step %{{.*}} {
! CHECK: %[[cast_temp:.*]] = fir.convert %[[temp:.*]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
! CHECK: %{{.*}} = fir.call @_FortranAioOutputAscii(%{{.*}}, %[[cast_temp]], %[[c10]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1

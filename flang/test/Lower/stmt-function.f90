! RUN: bbc -emit-fir -outline-intrinsics %s -o - | FileCheck %s

! Test statement function lowering

! Simple case
! CHECK-LABEL: func @_QPtest_stmt_0(%arg0: !fir.ref<f32>) -> f32
real function test_stmt_0(x)
  real :: x, func, arg
  func(arg) = arg + 0.123456

  ! CHECK-DAG: %[[x:.*]] = fir.load %arg0
  ! CHECK-DAG: %[[cst:.*]] = constant 1.234560e-01
  ! CHECK: %[[eval:.*]] = fir.addf %[[x]], %[[cst]]
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
  ! CHECK: %[[x1:.*]] = fir.call @_QPonly_once()
  ! Note: using -emit-fir, so the faked pass-by-reference is exposed
  ! CHECK: %[[x2:.*]] = fir.alloca f32
  ! CHECK: fir.store %[[x1]] to %[[x2]]
  ! CHECK: fir.addf %{{.*}}, %{{.*}}
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
  ! CHECK-DAG: %[[bmem:.*]] = fir.alloca f32 {name = "b"}
  ! CHECK-DAG: %[[res1:.*]] = fir.alloca f32 {name = "res1"}
  ! CHECK-DAG: %[[res2:.*]] = fir.alloca f32 {name = "res2"}

  b = 5

  ! CHECK-DAG: %[[cst_8:.*]] = constant 8.000000e+00
  ! CHECK-DAG: fir.store %[[cst_8]] to %[[tmp1:.*]] : !fir.ref<f32>
  ! CHECK-DAG: %[[foocall1:.*]] = fir.call @_QPfoo(%[[tmp1]])
  ! CHECK-DAG: %[[aload1:.*]] = fir.load %arg1
  ! CHECK: %[[add1:.*]] = fir.addf %[[aload1]], %[[foocall1]]
  ! CHECK: fir.store %[[add1]] to %[[res1]]
  res1 =  func1(8.)

  ! CHECK-DAG: %[[a2:.*]] = fir.load %arg1
  ! CHECK-DAG: %[[foocall2:.*]] = fir.call @_QPfoo(%arg0)
  ! CHECK-DAG: %[[add2:.*]] = fir.addf %[[a2]], %[[foocall2]]
  ! CHECK-DAG: %[[b:.*]] = fir.load %[[bmem]]
  ! CHECK: %[[add3:.*]] = fir.addf %[[add2]], %[[b]]
  ! CHECK: fir.store %[[add3]] to %[[res2]]
  res2 = func2(x)

  ! CHECK-DAG: %[[res12:.*]] = fir.load %[[res1]]
  ! CHECK-DAG: %[[res22:.*]] = fir.load %[[res2]]
  ! CHECK: = fir.addf %[[res12]], %[[res22]] : f32
  test_stmt_1 = res1 + res2
  ! CHECK: return %{{.*}} : f32
end function


! Test statement functions with no argument.
! Test that they are not pre-evaluated.
! CHECK-LABEL: func @_QPtest_stmt_no_args
real function test_stmt_no_args(x, y)
  func() = x + y
  ! CHECK: fir.addf
  a = func()
  ! CHECK: fir.call @_QPfoo_may_modify_xy
  call foo_may_modify_xy(x, y)
  ! CHECK: fir.addf
  ! CHECK: fir.addf
  test_stmt_no_args = func() + a
end function
  
! Test statement function with character arguments
! CHECK-LABEL: @_QPtest_stmt_character
integer function test_stmt_character(c, j)
   integer :: i, j, func, argj
   character(10) :: c, argc
   ! CHECK-DAG: %[[unboxed:.*]]:2 = fir.unboxchar %arg0 :
   ! CHECK-DAG: %[[c10:.*]] = constant 10 :
   ! CHECK: %[[c:.*]] = fir.emboxchar %[[unboxed]]#0, %[[c10]] 

   func(argc, argj) = len_trim(argc, 4) + argj
   ! CHECK: addi %{{.*}}, %{{.*}} : i
   test_stmt_character = func(c, j)
end function  

! issue #247
! CHECK-LABEL: @_QPbug247
subroutine bug247(r)
  I(R) = R
  ! CHECK: fir.call {{.*}}OutputInteger
  PRINT *, I(2.5)
  ! CHECK: fir.call {{.*}}EndIo
END subroutine bug247

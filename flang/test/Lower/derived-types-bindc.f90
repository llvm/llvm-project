! Test padding for BIND(C) derived types lowering for AIX target
! RUN: %flang_fc1 -emit-llvm %s -o - | FileCheck %s

! REQUIRES: target={{.+}}-aix{{.*}}

subroutine s1()
  use, intrinsic :: iso_c_binding
  type, bind(c) :: t0
    character(c_char) :: x1
    real(c_double) :: x2
  end type
  type(t0) :: xt0
! CHECK-DAG: %_QFs1Tt0 = type <{ [1 x i8], [3 x i8], double }>

  type, bind(c) :: t1
    integer(c_short) :: x1
    real(c_double) :: x2
  end type
  type(t1) :: xt1
! CHECK-DAG: %_QFs1Tt1 = type <{ i16, [2 x i8], double }>

  type, bind(c) :: t2
    integer(c_short) :: x1
    real(c_double) :: x2
    character(c_char) :: x3
  end type
  type(t2) :: xt2
! CHECK-DAG: %_QFs1Tt2 = type <{ i16, [2 x i8], double, [1 x i8], [3 x i8] }>

  type, bind(c) :: t3
    character(c_char) :: x1
    complex(c_double_complex) :: x2
  end type
  type(t3) :: xt3
! CHECK-DAG: %_QFs1Tt3 = type <{ [1 x i8], [3 x i8], { double, double } }>

  type, bind(c) :: t4
    integer(c_short) :: x1
    complex(c_double_complex) :: x2
    character(c_char) :: x3
  end type
  type(t4) :: xt4
! CHECK-DAG: %_QFs1Tt4 = type <{ i16, [2 x i8], { double, double }, [1 x i8], [3 x i8] }>
end subroutine s1

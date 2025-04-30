! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Check that correct opaque pointer-compatible IR is generated for
! pass-by-value arguments and struct-type return values, which typically
! appear in C binding interfaces.

! REQUIRES: x86-registered-target
! RUN: %flang -S -emit-flang-llvm %s -o %t
! RUN: FileCheck %s < %t

! CHECK: [[C_TYPE:%struct.c_type.*]] = type <{ double, double, double, double, double}>
! CHECK: [[C_FUNPTR:%struct.c_funptr.*]] = type <{ i64}>
! CHECK: define void @c_interface_byval_sub_({{.*}}, ptr byval([[C_FUNPTR]]) %_V_fp.arg)
! CHECK: call void (ptr, ptr, ...) @f90_c_f_procptr (ptr [[BSS:@\..+]], ptr {{.*}})
! CHECK: [[TMP:%.*]] = load i64, ptr [[BSS]]
! CHECK: [[TMP2:%.*]] = inttoptr i64 [[TMP]]
! CHECK: call void @c_interface_byval_sub_ ({{.*}}, ptr byval([[C_FUNPTR]]) [[TMP2]])
! CHECK: call void @c_function (ptr sret([[C_TYPE]]) {{%.*}}, ptr byval([[C_TYPE]]) %dur)
! CHECK: declare void @c_function(ptr sret([[C_TYPE]]), ptr byval([[C_TYPE]]))

module c_interface
  use, intrinsic :: iso_c_binding
  type, bind(c) :: c_type
     real(kind = c_double) :: year = 0, month = 0, day = 0, hour = 0, minute = 0
  end type
  interface
    type(c_type) function c_function(dur) bind(c)
      use iso_c_binding
      import :: c_type
      type(c_type), value :: dur
    end function
  end interface
contains
  ! Reproducer from https://github.com/flang-compiler/flang/issues/1419.
  subroutine byval_sub(x1, x2, x3, x4, x5, x6, fp)
    implicit none
    integer(c_int), intent(in), value :: x1
    integer(c_int), intent(in), value :: x2
    integer(c_int), intent(in), value :: x3
    integer(c_int), intent(in), value :: x4
    integer(c_int), intent(in), value :: x5
    integer(c_int), intent(in), value :: x6
    type(c_funptr), intent(in), value :: fp
  end subroutine
end module

module test
  use c_interface
contains
  function ss(dur) result(res)
    implicit none
    integer(c_int), parameter :: x = 42
    type(c_funptr) :: fp
    type(c_type), intent(in) :: dur
    type(c_type) :: res
    procedure(byval_sub), pointer :: proc
    call c_f_procpointer(fp, proc)
    call byval_sub(x, x, x, x, x, x, fp)
    res = c_function(dur)
  end function
end module

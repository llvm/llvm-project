! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! https://github.com/flang-compiler/flang/issues/320
! https://github.com/flang-compiler/flang/issues/1407
!
! The dllimport_test.f90 relies on 'dllimport_test.f90_mod.f90' external module
! to test the presence of 'dllimport' storage class in LLVM IR
! for external module and opaque type.
!
! REQUIRES: system-windows
!
! RUN: %flang -S -emit-llvm -S -emit-llvm %s_mod.f90 %s
! RUN: cat dllimport_test.ll | FileCheck %s
! CHECK: %structdllimport_module__t_type__td_ = type opaque
! CHECK: @_dllimport_module_10_ = external dllimport global
! CHECK: @dllimport_module__t_type__td_ = external dllimport global
program h_main
  use dllimport_module
  implicit none

  call foobar(array)
end program

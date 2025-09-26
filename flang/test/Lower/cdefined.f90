! RUN: bbc -emit-hlfir -o - %s | FileCheck %s
! Ensure that CDEFINED variable has external (default) linkage and that
! it doesn't have an initializer
module m
  use iso_c_binding
  integer(c_int), bind(C, name='c_global', CDEFINED) :: c  = 42
  ! CHECK: fir.global @c_global : i32
  ! CHECK-NOT: fir.zero_bits 
end

! RUN: bbc -emit-fir -o - %s | FileCheck %s

module bind_c_type
  use, intrinsic :: iso_c_binding

  type, bind(C) :: t
    type(c_ptr) :: tcptr = C_NULL_PTR
  end type
end module

! CHECK-LABEL: _QMbind_c_typeE.di.t.tcptr

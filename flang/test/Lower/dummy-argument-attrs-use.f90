! The caller only has the module interface. Intent must still appear on the
! private func.func declaration, with no callee body to inspect.
!
! RUN: rm -fr %t && mkdir -p %t && cd %t
! RUN: bbc -emit-hlfir %S/dummy-argument-attrs-module.f90
! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

program p
  use m
  implicit none
  double precision :: a, b
  a = 1.0d0
  call twice(a, b)
end program p

! CHECK: func.func private @_QMmPtwice(!fir.ref<f64> {fir.fortran_attrs = #fir.var_attrs<intent_in>, fir.read_only}, !fir.ref<f64> {fir.fortran_attrs = #fir.var_attrs<intent_out>})

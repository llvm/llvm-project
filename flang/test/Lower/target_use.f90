! RUN: bbc -emit-fir %S/target_definition.f90
! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Test TARGET attributes on a declaration of a global symbol.
! CHECK: fir.global @_QMtarget_modEx target : f32
real function test()
  use target_mod
  test = x
end function test

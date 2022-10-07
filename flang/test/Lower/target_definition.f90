! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Test TARGET attributes on a definition of a global symbol.
! CHECK: fir.global @_QMtarget_modEx target : f32 {
! CHECK:   %[[init:.*]] = fir.undefined f32
! CHECK:   fir.has_value %[[init]] : f32
! CHECK: }

module target_mod
  real, target :: x
end module target_mod

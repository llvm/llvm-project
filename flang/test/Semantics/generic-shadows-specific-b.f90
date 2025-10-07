! Check that expected code produced with no crash.
subroutine reproducer()
  use m2
  use m3
  character x
  x = f()
end

! RUN: %flang_fc1 -fsyntax-only %S/Inputs/generic-shadows-specific-a.f90
! RUN: bbc -emit-fir -o - %s | FileCheck %s

! CHECK-LABEL: func.func @_QPreproducer
! CHECK: fir.call @_QMm1Pf

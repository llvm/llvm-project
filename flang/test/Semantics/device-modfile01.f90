! Inputs/device-side-modules-a.mod records a deliberately wrong checksum for its
! dependency on the intrinsic module iso_fortran_env.

! Without offloading the recorded checksum is enforced and the wrong-checksum
! dependency is rejected.
! RUN: not %flang_fc1 -fsyntax-only -I%S/Inputs %s 2>&1 | FileCheck %s --check-prefix=HOST
! HOST: error: Cannot use module file for module 'iso_fortran_env': File is not the right module file for 'iso_fortran_env'

! When compiling for the device, the intrinsic module checksum is ignored.
! RUN: %flang_fc1 -fsyntax-only -foffload-device -I%S/Inputs %s

module device_modfile01
  use device_modfile01_a, only: x
end module

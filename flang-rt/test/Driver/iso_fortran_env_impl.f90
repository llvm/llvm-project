! Verify that iso_fortran_env constants from the C++ runtime
! (iso_fortran_env_impl.cpp) match the Fortran module definitions.

! UNSUPPORTED: offload-cuda
! REQUIRES: fortran-modules

! RUN: %flang %isysroot -L"%libdir" %s -o %t
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t | FileCheck %s

program check_iso_env
  use iso_fortran_env, only: int8, int16, int32, int64, real32, real64

  ! CHECK: PASS int8 1
  print '("PASS int8 ",I0)', int8
  ! CHECK: PASS int16 2
  print '("PASS int16 ",I0)', int16
  ! CHECK: PASS int32 4
  print '("PASS int32 ",I0)', int32
  ! CHECK: PASS int64 8
  print '("PASS int64 ",I0)', int64
  ! CHECK: PASS real32 4
  print '("PASS real32 ",I0)', real32
  ! CHECK: PASS real64 8
  print '("PASS real64 ",I0)', real64
end program

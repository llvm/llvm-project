! Check that the macros that give the version number are set properly

!CHECK: flang_major = {{[1-9][0-9]*$}}
!CHECK: flang_minor = {{[0-9]+$}}
!CHECK: flang_patchlevel = {{[0-9]+$}}
!RUN: %f18 -E %s | FileCheck  --ignore-case %s

  
integer, parameter :: flang_major = __flang_major__
integer, parameter :: flang_minor = __flang_minor__
integer, parameter :: flang_patchlevel = __flang_patchlevel__

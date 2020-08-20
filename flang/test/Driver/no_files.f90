! RUN: %f18 < %S/Inputs/hello.f90 | FileCheck %s


! CHECK: Enter Fortran source
! CHECK: Use EOF character (^D) to end file

! CHECK: Parse tree comprises {{.*}} objects and occupies {{.*}} total bytes
! CHECK: PROGRAM hello
! CHECK:  WRITE (*, *) "hello world"
! CHECK: END PROGRAM hello

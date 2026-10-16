! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck %s

! A module that this compilation unit does not define is described as a
! declaration. Nothing here compiles iso_fortran_env, so its DIModule must carry
! no scope, file or line, even though using it materializes named constants of
! its own in this unit.

program p
  use iso_fortran_env
  implicit none
  print *, 'hello'
end program p

! CHECK: !DIModule(scope: null, name: "iso_fortran_env", isDecl: true)

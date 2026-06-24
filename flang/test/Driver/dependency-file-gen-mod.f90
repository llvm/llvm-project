! Module files used via `use` are real build dependencies. -MD/-MMD run a full
! compile, so the .mod opened during semantics is recorded in the output. Plain
! -M/-MM is prescan-only and does not resolve modules, so it lists includes only.

! RUN: rm -rf %t && split-file %s %t
! RUN: %flang -fsyntax-only %t/mymod.f90 -J %t

! -MD: the output lists the user module plus the intrinsic modules opened during
! semantics (compiler-provided __fortran_builtins.mod / __fortran_type_info.mod).
! RUN: %flang -MD -c %t/main.f90 -J %t -o %t/main.o
! RUN: FileCheck %s --input-file=%t/main.d --check-prefix=MD
! MD: main.o:
! MD-DAG: main.f90
! MD-DAG: mymod.mod
! MD-DAG: __fortran_builtins.mod
! MD-DAG: __fortran_type_info.mod

! -M stays prescan-only: no module dependency.
! RUN: %flang -M %t/main.f90 -J %t 2>&1 | FileCheck %s --check-prefix=M
! M: main.o:
! M-NOT: mymod.mod

!--- mymod.f90
module mymod
  integer :: v = 1
end module mymod

!--- main.f90
program main
  use mymod
  implicit none
  print *, v
end program main

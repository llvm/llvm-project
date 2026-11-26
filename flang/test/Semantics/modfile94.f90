! Verify that the module file of a module whose ambiguous names involve a
! module re-exported by a plain USE refers to every module that it names, so
! that a consumer that has only that module file can read it, also when the
! module file is hermetic.

! RUN: rm -rf %t && split-file %s %t && mkdir %t/build %t/use %t/hermetic
! RUN: %flang_fc1 -fsyntax-only -J%t/build %t/origin.f90
! RUN: %flang_fc1 -fsyntax-only -J%t/build %t/other.f90
! RUN: %flang_fc1 -fsyntax-only -J%t/build %t/facade.f90
! RUN: FileCheck --check-prefix=NEED --input-file=%t/build/mf94_facade.mod %s
! RUN: %flang_fc1 -fsyntax-only -fhermetic-module-files -J%t/build %t/facade.f90
! RUN: cp %t/build/mf94_facade.mod %t/hermetic
! RUN: %flang_fc1 -fsyntax-only -J%t/hermetic %t/q.f90

! NEED-DAG: !need${{.*}} n mf94_origin
! NEED-DAG: !need${{.*}} n mf94_other

!--- origin.f90
module mf94_origin
  structure /s/
    integer :: i
  end structure
  integer, parameter :: x = 1
end module

!--- other.f90
module mf94_other
  integer, parameter :: y = 2
end module

!--- facade.f90
module mf94_facade
  use mf94_origin, z => x
  use mf94_other, z => y
end module

!--- q.f90
program q
  use mf94_facade
end program

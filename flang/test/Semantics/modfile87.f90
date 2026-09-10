! RUN: split-file %s %t
! RUN: %flang_fc1 -fsyntax-only -J%t %t/m.f90
! RUN: %flang_fc1 -fsyntax-only -pedantic -J%t %t/s.f90 2>&1 | FileCheck %s

! When a module and its submodule are compiled in separate
! invocations (so that the submodule's parent module scope is read back from
! the .mod file), the "missing MODULE prefix" portability warning for the
! submodule's subprogram must still be emitted.

!--- m.f90
module modfile87m
  interface
    module subroutine inside_one()
    end subroutine
  end interface
end module

!--- s.f90
submodule (modfile87m) modfile87s
contains
  subroutine inside_one()
  end subroutine
end submodule

!CHECK: portability: Subprogram 'inside_one' in this submodule is missing the MODULE prefix

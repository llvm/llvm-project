! RUN: split-file %s %t
! RUN: %flang_fc1 -fsyntax-only -module-dir %t %t/m.f90
! RUN: not %flang_fc1 -fsyntax-only -module-dir %t %t/use.f90 2>&1 | FileCheck %s
! F2023 19.2 p2: the binding label of a BIND(C) variable that is use-associated
! from a module file conflicts with an external procedure declared in the using
! scope.  The error is reported at the declaration in the file being compiled.

!--- m.f90
module global05m
  use, intrinsic :: iso_c_binding, only : c_int
  integer(c_int), bind(c, name='clash') :: counter
end

!--- use.f90
subroutine caller
  use global05m
  interface
    !CHECK: use.f90:{{[0-9]+}}:{{[0-9]+}}: error: Two entities have the same global name 'clash'
    !CHECK: global05m.mod:{{[0-9]+}}:{{[0-9]+}}: Conflicting declaration
    subroutine clash() bind(c)
    end subroutine
  end interface
  counter = 1
  call clash
end

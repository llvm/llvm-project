! RUN: split-file %s %t
! RUN: %flang_fc1 -fsyntax-only -x cuda -module-dir %t %t/m.cuf
! RUN: not %flang_fc1 -fsyntax-only -fopenacc -module-dir %t %t/use.f90 2>&1 | FileCheck %s

!--- m.cuf
module modfile84m
  real, device :: d
contains
  attributes(device) subroutine s()
  end subroutine
end module

!--- use.f90
use modfile84m
end

! CHECK: error: Cannot use module file for module 'modfile84m': CUDA is not enabled, but '{{.*modfile84m.mod}}' defines CUDA symbols

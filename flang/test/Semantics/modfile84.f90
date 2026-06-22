! RUN: split-file %s %t
! RUN: %flang_fc1 -fsyntax-only -x cuda -module-dir %t %t/m.cuf
! RUN: not %flang_fc1 -fsyntax-only -fopenacc -module-dir %t %t/use.f90 2>&1 | FileCheck %s

!--- m.cuf
module m
  real, device :: d
contains
  attributes(device) subroutine s()
  end subroutine
end module

!--- use.f90
use m
end

! CHECK: error: Cannot use module file for module 'm': CUDA is not enabled, but '{{.*m.mod}}' defines CUDA symbols

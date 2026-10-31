! RUN: split-file %s %t
! RUN: %flang_fc1 -fsyntax-only -x cuda -module-dir %t %t/m.cuf
! RUN: not %flang_fc1 -fsyntax-only -fopenacc -module-dir %t %t/use.f90 2>&1 | FileCheck %s
! RUN: %flang_fc1 -fsyntax-only -x cuda -module-dir %t %t/m2.cuf
! RUN: %flang_fc1 -fsyntax-only -fopenacc -module-dir %t %t/use2.f90
! RUN: %flang_fc1 -fsyntax-only -x cuda -module-dir %t %t/m3.cuf
! RUN: not %flang_fc1 -fsyntax-only -fopenacc -module-dir %t %t/use3.f90 2>&1 | FileCheck --check-prefix=CHECK-M3 %s

!--- m.cuf
module modfile84m
  real, device :: d
contains
  attributes(device) subroutine s()
  end subroutine
end module

!--- m2.cuf
module modfile84m2
contains
  attributes(device) subroutine s()
  end subroutine
end module

!--- m3.cuf
module modfile84m3
  type dt
    real, device, pointer :: dp
  end type

  type(dt) :: d ! Should trigger the error
end module

!--- use.f90
use modfile84m
end

!--- use2.f90
use modfile84m2
end

!--- use3.f90
use modfile84m3
end

! CHECK: error: Cannot use module file for module 'modfile84m': CUDA is not enabled, but '{{.*modfile84m.mod}}' defines CUDA symbols
! CHECK-M3: error: Cannot use module file for module 'modfile84m3': CUDA is not enabled, but '{{.*modfile84m3.mod}}' defines CUDA symbols

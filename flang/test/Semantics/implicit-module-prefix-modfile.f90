! RUN: split-file %s %t
! RUN: %flang_fc1 -fsyntax-only -J%t %t/m.f90
! RUN: %flang_fc1 -fsyntax-only -fimplicit-module-prefix -J%t %t/s.f90 2>&1 | FileCheck %s --allow-empty --check-prefix=SILENT
! RUN: %flang_fc1 -fsyntax-only -fimplicit-module-prefix -Wimplicit-module-prefix -J%t %t/s.f90 2>&1 | FileCheck %s --check-prefix=REPAIR
! RUN: %flang_fc1 -fsyntax-only -fimplicit-module-prefix -pedantic -J%t %t/s.f90 2>&1 | FileCheck %s --check-prefix=REPAIR
! RUN: %flang_fc1 -fsyntax-only -fimplicit-module-prefix -pedantic -J%t %t/t.f90 2>&1 | FileCheck %s --allow-empty --check-prefix=IMPORT

! A repair in current source must be reported even when the parent comes
! from a .mod file. Reading the repaired .smod must not repeat the warning.

!--- m.f90
module implicit_prefix_parent
  interface
    module subroutine implementation()
    end subroutine
  end interface
end module

!--- s.f90
submodule (implicit_prefix_parent) implicit_prefix_child
  interface
    module subroutine next_implementation()
    end subroutine
  end interface
contains
  subroutine implementation()
  end subroutine
end submodule

!--- t.f90
submodule (implicit_prefix_parent:implicit_prefix_child) implicit_prefix_grandchild
contains
  module subroutine next_implementation()
  end subroutine
end submodule

! SILENT-NOT: warning:
! SILENT-NOT: portability:
! REPAIR: portability: Assuming a missing MODULE prefix on 'implementation' to repair the separate module procedure interface 'implicit_prefix_parent:implementation' [-Wimplicit-module-prefix]
! REPAIR-NOT: missing the MODULE prefix
! REPAIR-NOT: Assuming a missing MODULE prefix
! IMPORT-NOT: warning:
! IMPORT-NOT: portability:

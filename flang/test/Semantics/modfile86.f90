! RUN: split-file %s %t
! RUN: %flang_fc1 -fsyntax-only -J%t %t/a.f90
! RUN: %flang_fc1 -fsyntax-only -J%t %t/b.f90
! RUN: %flang_fc1 -fsyntax-only -pedantic -J%t %t/c.f90 2>&1 | FileCheck --allow-empty %s

! Compiling a submodule of a submodule ("b") must not
! resurface "b"'s own missing-MODULE-prefix portability warning when "b" is
! re-read from its .smod file as a dependency of "c".

!--- a.f90
module modfile86a
  interface
    module subroutine inside_one()
    end subroutine
  end interface
end module

!--- b.f90
submodule (modfile86a) modfile86b
  interface
    module subroutine inside_two()
    end subroutine
  end interface
contains
  subroutine inside_one()
  end subroutine
end submodule

!--- c.f90
submodule (modfile86a:modfile86b) modfile86c
contains
  module subroutine inside_two()
  end subroutine
end submodule

!CHECK-NOT: portability

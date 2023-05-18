! Test component name resolution with nested legacy DEC structures.
!RUN: %flang_fc1 -fdebug-dump-symbols %s | FileCheck %s

  structure /a/
    integer :: a_first_comp
    structure /b/ b1, b2(100)
      integer :: i
    end structure
    structure /c/ z
      integer :: i
      structure /d/ d1, d2(10)
        real :: x
      end structure
    end structure
    integer :: a_last_comp
  end structure
end
! CHECK:    /a/: DerivedType sequence components: a_first_comp,b1,b2,z,a_last_comp
! CHECK:    /b/: DerivedType sequence components: i
! CHECK:    /c/: DerivedType sequence components: i,d1,d2
! CHECK:    /d/: DerivedType sequence components: x

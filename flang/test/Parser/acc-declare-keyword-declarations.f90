! RUN: split-file %s %t
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/function.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/function.f
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/subroutine.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/subroutine.f
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/character.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/character.f
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/integer.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/integer.f
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/array.f90

! These are declarations in unnamed main programs, not procedure headers.
! Fortran keywords are not reserved (F2023 5.5.2, R801).

!--- function.f90
!$acc declare create(x)
real functionx
real x
x = 1.0
end

!--- function.f
c$acc declare create(x)
      real functionx
      real x
      x = 1.0
      end

!--- subroutine.f90
!$acc declare create(x)
real subroutinex
real x
x = 1.0
end

!--- subroutine.f
c$acc declare create(x)
      real subroutinex
      real x
      x = 1.0
      end

!--- character.f90
!$acc declare create(x)
character(len=8) functionname
real x
x = 1.0
end

!--- character.f
c$acc declare create(x)
      character(len=8) functionname
      real x
      x = 1.0
      end

!--- integer.f90
!$acc declare create(x)
integer subroutinecount
real x
x = 1.0
end

!--- integer.f
c$acc declare create(x)
      integer subroutinecount
      real x
      x = 1.0
      end

!--- array.f90
!$acc declare create(x)
real functionx(n)
integer, parameter :: n = 2
real x
functionx = 1.0
x = functionx(1)
end

! RUN: split-file %s %t
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/first-do.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/first-block.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/first-select.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/first-associate.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/first-do-while.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/first-do-concurrent.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/first-labeled-do.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/after-do.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/after-block.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/after-select.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/after-associate.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/after-do-while.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/after-do-concurrent.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/after-labeled-do.f90

! Construct names starting with SUBROUTINE must not identify a procedure
! header after DECLARE, including after another program unit. The DECLARE
! belongs to the specification part of the unnamed main program.

!--- first-do.f90
!$acc declare create(x)
subroutine_loop: do i = 1, 2
end do subroutine_loop
end

!--- first-block.f90
!$acc declare create(x)
subroutine_block: block
end block subroutine_block
end

!--- first-select.f90
!$acc declare create(x)
subroutine_select: select case (1)
case default
end select subroutine_select
end

!--- first-associate.f90
!$acc declare create(x)
subroutine_associate: associate (a => 1)
end associate subroutine_associate
end

!--- first-do-while.f90
!$acc declare create(x)
subroutine_while: do while (.false.)
end do subroutine_while
end

!--- first-do-concurrent.f90
!$acc declare create(x)
subroutine_concurrent: do concurrent (i = 1:2)
end do subroutine_concurrent
end

!--- first-labeled-do.f90
!$acc declare create(x)
10 subroutine_labeled: do i = 1, 2
end do subroutine_labeled
end

!--- after-do.f90
subroutine previous()
end subroutine previous

!$acc declare create(x)
subroutine_loop: do i = 1, 2
end do subroutine_loop
end

!--- after-block.f90
subroutine previous()
end subroutine previous

!$acc declare create(x)
subroutine_block: block
end block subroutine_block
end

!--- after-select.f90
subroutine previous()
end subroutine previous

!$acc declare create(x)
subroutine_select: select case (1)
case default
end select subroutine_select
end

!--- after-associate.f90
subroutine previous()
end subroutine previous

!$acc declare create(x)
subroutine_associate: associate (a => 1)
end associate subroutine_associate
end

!--- after-do-while.f90
subroutine previous()
end subroutine previous

!$acc declare create(x)
subroutine_while: do while (.false.)
end do subroutine_while
end

!--- after-do-concurrent.f90
subroutine previous()
end subroutine previous

!$acc declare create(x)
subroutine_concurrent: do concurrent (i = 1:2)
end do subroutine_concurrent
end

!--- after-labeled-do.f90
subroutine previous()
end subroutine previous

!$acc declare create(x)
10 subroutine_labeled: do i = 1, 2
end do subroutine_labeled
end

! RUN: %python %S/test_errors.py %s %flang_fc1
! Every labeled statement records its own source position, so a diagnostic
! about the label is reported on that statement.  The position of a labeled
! END statement of a program unit is the case worth pinning: label analysis
! visits it in advance, before the statement visitor reaches it, so the
! position has to be supplied explicitly rather than taken from the visitor's
! current position.  Without that, the diagnostic below has no location at all.
!
! `write(*,fmt=L)` names L as a format; every statement here is something other
! than a FORMAT statement, so each one is reported.

! Labeled END statement of the first program unit in the file.
subroutine end_first_unit()
  write(*,fmt=53)
!ERROR: '53' not a FORMAT
53 end subroutine

subroutine construct_stmts(n)
  integer :: n

  write(*,fmt=10)
  write(*,fmt=11)
  write(*,fmt=12)
  write(*,fmt=40)

  ! Statement that begins a construct.
  !ERROR: '10' not a FORMAT
10 if (n > 0) then
  end if

  !ERROR: '11' not a FORMAT
11 do n = 1, 2
  end do

  !ERROR: '12' not a FORMAT
12 select case (n)
  case default
  end select

  ! END IF and END SELECT: the label is not in the last part or case.  The
  ! reference sits inside the construct, where naming its END statement is
  ! permitted, so that only the position is under test here.
  if (n > 0) then
    write(*,fmt=20)
  !ERROR: '20' not a FORMAT
20 end if

  select case (n)
  case default
    write(*,fmt=21)
  !ERROR: '21' not a FORMAT
21 end select

  ! Statement that ends a construct.
  do n = 1, 2
    write(*,fmt=30)
  !ERROR: '30' not a FORMAT
30 end do

  ! Ordinary executable statement.
  !ERROR: '40' not a FORMAT
40 continue
end subroutine

! Labeled END statement of a subroutine.
subroutine end_subroutine()
  write(*,fmt=50)
!ERROR: '50' not a FORMAT
50 end subroutine

! Labeled END statement of a function.
function end_function()
  integer :: end_function
  end_function = 0
  write(*,fmt=51)
!ERROR: '51' not a FORMAT
51 end function

! Labeled END statement of the main program.
program end_program
  write(*,fmt=52)
!ERROR: '52' not a FORMAT
52 end program

! A labeled program-unit END as the terminal statement of a labeled DO
! construct.
subroutine do_terminal()
  integer :: i
  do 54 i = 1, 3
    print *, i
!ERROR: This statement cannot terminate the DO loop
54 end subroutine

! Labeled END PROCEDURE statement of a separate module subprogram.
module m_sub
  interface
    module subroutine sub()
    end subroutine
  end interface
end module

submodule (m_sub) m_sub_impl
contains
  module procedure sub
    write(*,fmt=55)
!ERROR: '55' not a FORMAT
55 end procedure
end submodule

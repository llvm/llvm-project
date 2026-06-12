!RUN: %python %S/../Semantics/test_errors.py %s %flang_fc1 -pedantic -Werror
blockdata
endblockdata
block data
end blockdata
block data
endblock data

subroutine one(p)
  doubleprecision d1
  double precision d2
  enum, bind(c)
    enumerator :: e1=1
  endenum
  enum, bind(c)
    enumerator :: e2=2
  end enum
  class(*), pointer :: p
  integer assigned

  d1 = 0.
  d2 = 0.
  associate(s=>d1)
  endassociate
  associate(s=>d2)
  end associate
  block
  endblock
  block
  end block
  critical
  endcritical
  critical
  end critical
  do
  enddo
  do
  end do
  endfile(1)
  end file(1)
  if (.false.) then
  elseif (.false.) then
  else if (.false.) then
  endif
  if (.false.) then
  end if
  where ([.false.])
  elsewhere ([.false.])
  else where ([.false.])
  endwhere
  where ([.false.])
  end where
  forall(j=1:2)
  endforall
  forall(j=1:2)
  end forall
  selectcase(1)
  case(1)
  endselect
  select case(1)
  case(1)
  end select
  selecttype(p)
  type is(doubleprecision)
  endselect
  select type(p)
  type is(double precision)
  end select
  !PORTABILITY: deprecated usage
  assign 10 to assigned
  goto 10
10 go to 20
  !PORTABILITY: deprecated usage
20 goto assigned
  !PORTABILITY: deprecated usage
  go to assigned
endsubroutine

subroutine two(x,y)
  use iso_fortran_env, only: team_type
  real, intent(inout) :: x
  real, intent(in out) :: y
  type(team_type) t
  change team(t)
  endteam
  change team(t)
  end team
end subroutine
subroutine three
endsubroutine three
subroutine four
end subroutine four

function f1()
  f1 = 0.
endfunction
function f2()
  f2 = 0.
end function
function f3()
  f3 = 0.
endfunction f3
function f4()
  f4 = 0.
end function f4
module m1
  interface
    module subroutine p1
    endsubroutine
    module subroutine p2
    end subroutine
    module subroutine p3
    endsubroutine p3
    module subroutine p4
    end subroutine p4
  endinterface
  interface
  end interface
  interface g1
  endinterface g1
  interface g2
  end interface g2
endmodule
module m2
end module
module m3
endmodule m3
module m4
end module m4
submodule(m1) sm1
 contains
  module procedure p1
  endprocedure
endsubmodule
submodule(m1) sm2
 contains
  module procedure p2
  end procedure
end submodule
submodule(m1) sm3
 contains
  module procedure p3
  endprocedure p3
endsubmodule sm3
submodule(m1) sm4
 contains
  module procedure p4
  end procedure p4
end submodule sm4

program p
  !PORTABILITY: missing space
  realprogrammersusefortran
  !PORTABILITY: missing space
  commonprogrammersusefortran
endprogram p

! RUN: %python %S/test_errors.py %s %flang_fc1
! Tests for the last sentence of C1128:
!A variable-name that is not permitted to appear in a variable definition
!context shall not appear in a LOCAL or LOCAL_INIT locality-spec.

subroutine s1(arg)
  real, intent(in) :: arg

  ! This is not OK because "arg" is "intent(in)"
!ERROR: INTENT IN argument 'arg' not allowed in a locality-spec
  do concurrent (i=1:5) local(arg)
  end do
end subroutine s1

subroutine s2(arg)
  real, value, intent(in) :: arg

  ! This is not OK even though "arg" has the "value" attribute.  C1128
  ! explicitly excludes dummy arguments of INTENT(IN)
!ERROR: INTENT IN argument 'arg' not allowed in a locality-spec
  do concurrent (i=1:5) local(arg)
  end do
end subroutine s2

module m3
  real, protected :: prot
  real var

  contains
    subroutine sub()
      ! C857 This is OK because of the "protected" attribute only applies to
      ! accesses outside the module
      do concurrent (i=1:5) local(prot)
      end do
    end subroutine sub
endmodule m3

subroutine s4()
  use m3

  ! C857 This is not OK because of the "protected" attribute
!ERROR: 'prot' may not appear in a locality-spec because it is not definable
!BECAUSE: 'prot' is protected in this scope
  do concurrent (i=1:5) local(prot)
  end do

  ! C857 This is OK because of there's no "protected" attribute
  do concurrent (i=1:5) local(var)
  end do
end subroutine s4

subroutine s5()
  real :: a, b, c, d, e

  associate (a => b + c, d => e)
    b = 3.0
    ! C1101 This is OK because 'd' is associated with a variable
    do concurrent (i=1:5) local(d)
    end do

    ! C1101 This is not OK because 'a' is not associated with a variable
!ERROR: 'a' may not appear in a locality-spec because it is not definable
!BECAUSE: 'a' is construct associated with an expression
    do concurrent (i=1:5) local(a)
    end do
  end associate
end subroutine s5

subroutine s6()
  type point
    real :: x, y
  end type point

  type, extends(point) :: color_point
    integer :: color
  end type color_point

  type(point), target :: c, d
  class(point), pointer :: p_or_c

  p_or_c => c
  select type ( a => p_or_c )
  type is ( point )
    ! C1158 This is OK because 'a' is associated with a variable
    do concurrent (i=1:5) local(a)
    end do
  end select

  select type ( a => func() )
  type is ( point )
    ! C1158 This is OK because 'a' is associated with a variable
    do concurrent (i=1:5) local(a)
    end do
  end select

  select type ( a => (func()) )
  type is ( point )
    ! C1158 This is not OK because 'a' is not associated with a variable
!ERROR: 'a' may not appear in a locality-spec because it is not definable
!BECAUSE: 'a' is construct associated with an expression
    do concurrent (i=1:5) local(a)
    end do
  end select

  contains
    function func()
      class(point), pointer :: func
      func => c
    end function func
end subroutine s6

module m4
  real, protected :: prot
  real var
endmodule m4

pure subroutine s7()
  use m4

  ! C1594 This is not OK because we're in a PURE subroutine
!ERROR: 'var' may not appear in a locality-spec because it is not definable
!BECAUSE: 'var' may not be defined in pure subprogram 's7' because it is USE-associated
  do concurrent (i=1:5) local(var)
  end do
end subroutine s7

subroutine s8()
  integer, parameter :: iconst = 343

!ERROR: 'iconst' may not appear in a locality-spec because it is not definable
!BECAUSE: 'iconst' is not a variable
  do concurrent (i=1:5) local(iconst)
  end do
end subroutine s8

! A variable declared in a BLOCK may appear in a locality-spec of a
! DO CONCURRENT in that BLOCK or in a nested BLOCK.
subroutine block_local_locality(n)
  integer :: n, j
  block
    integer :: t1, t2, t3, t4, t5
    t2 = 1
    t3 = 0
    do concurrent (j=1:n) local(t1) local_init(t2) reduce(+:t3) shared(t4)
      t1 = j + t2
      t3 = t3 + t1
    end do
    t4 = t3
    block
      do concurrent (j=1:n) local(t5)
        t5 = j
      end do
    end block
  end block
end subroutine

! A nested BLOCK that reaches a variable of the enclosing BLOCK through
! host association (asynchronous I/O, or an ASYNCHRONOUS or VOLATILE
! statement) may name it in a locality-spec.
subroutine block_host_assoc_locality(n)
  integer :: n, j
  block
    integer :: x1, x2, x3
    block
      do concurrent (j=1:n) local(x1)
        x1 = j
      end do
      write(10, asynchronous='yes') x1
    end block
    block
      asynchronous :: x2
      do concurrent (j=1:n) shared(x2)
      end do
    end block
    block
      volatile :: x3
      do concurrent (j=1:n) local(x3)
        x3 = j
      end do
    end block
  end block
end subroutine

! A BLOCK entity that cannot be a variable is still rejected.
subroutine block_local_not_variable(n)
  integer :: n, j
  block
    integer, external :: f
!ERROR: 'f' may not appear in a locality-spec because it is not definable
!BECAUSE: 'f' is not a variable
    do concurrent (j=1:n) local(f)
    end do
!ERROR: 'f' may not appear in a locality-spec because it is not definable
!BECAUSE: 'f' is not a variable
    do concurrent (j=1:n) local_init(f)
    end do
!ERROR: 'f' may not appear in a locality-spec because it is not definable
!BECAUSE: 'f' is not a variable
    do concurrent (j=1:n) reduce(+:f)
    end do
!ERROR: The name 'f' must be a variable to appear in a locality-spec
    do concurrent (j=1:n) shared(f)
    end do
  end block
end subroutine

! Other kinds of BLOCK entity in a locality-spec: a POINTER, an entity with
! no attributes that could otherwise later be a function, and an implicitly
! typed name that the locality-spec declares.
subroutine block_local_entity_kinds(n)
  integer :: n, j
  block
    integer, pointer :: p
    integer :: f
    do concurrent (j=1:n) local(p, f)
    end do
    do concurrent (j=1:n) local(q)
      q = j
    end do
  end block
end subroutine

! The usual locality-spec checks still apply to BLOCK entities, and a name
! in a locality-spec is a variable for the rest of the BLOCK.
subroutine block_local_entity_errors(n)
  integer :: n, j
  block
    integer, allocatable :: a
    integer, parameter :: k = 1
    integer :: f
!ERROR: ALLOCATABLE variable 'a' not allowed in a LOCAL locality-spec
    do concurrent (j=1:n) local(a)
    end do
!ERROR: 'k' may not appear in a locality-spec because it is not definable
!BECAUSE: 'k' is not a variable
    do concurrent (j=1:n) local(k)
    end do
!ERROR: The name 'k' must be a variable to appear in a locality-spec
    do concurrent (j=1:n) shared(k)
    end do
    do concurrent (j=1:n) local(f)
    end do
!ERROR: 'f' is not a callable procedure
    j = f(1)
  end block
end subroutine

subroutine block_local_implicit_none(n)
  implicit none
  integer :: n, j
  block
!ERROR: No explicit type declared for 'q'
    do concurrent (j=1:n) local(q)
    end do
  end block
end subroutine

! REDUCE is still rejected for an ASYNCHRONOUS or VOLATILE host-associated
! name in a nested BLOCK.
subroutine block_host_assoc_reduce(n)
  integer :: n, j
  block
    integer :: y1, y2
    y1 = 0
    y2 = 0
    block
      asynchronous :: y1
!ERROR: ASYNCHRONOUS variable 'y1' not allowed in a REDUCE locality-spec
      do concurrent (j=1:n) reduce(+:y1)
        y1 = y1 + j
      end do
    end block
    block
      volatile :: y2
!ERROR: VOLATILE variable 'y2' not allowed in a REDUCE locality-spec
      do concurrent (j=1:n) reduce(+:y2)
        y2 = y2 + j
      end do
    end block
  end block
end subroutine

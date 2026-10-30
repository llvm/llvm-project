! RUN: %python %S/test_errors.py %s %flang_fc1
! F2023 19.2 p2: an external procedure is a global entity wherever it is
! declared, so its global identifier (its name, or its binding label) may not
! be the binding label of a BIND(C) variable declared in another scope.
module m
  use, intrinsic :: iso_c_binding, only : c_int
  integer(c_int), bind(c, name='clash1') :: v1
  integer(c_int), bind(c, name='clash2') :: v2
  integer(c_int), bind(c, name='clash3') :: v3
  integer(c_int), bind(c, name='clash4') :: v4
  integer(c_int), bind(c, name='clash5') :: v5
  integer(c_int), bind(c, name='ok1') :: w1
  integer(c_int), bind(c, name='ok2') :: w2
  integer(c_int), bind(c, name='ok3') :: w3
  integer(c_int), bind(c, name='ok4') :: w4
  integer(c_int), bind(c, name='ok5') :: w5
end

! An external procedure declared in the specification part of a module.
module m2
  integer, bind(c, name='xxxyy') :: p1
  !ERROR: Two entities have the same global name 'xxxyy'
  procedure(real) :: xxxyy
end

! An interface body in a subprogram.
subroutine s1
  use m
  interface
    !ERROR: Two entities have the same global name 'clash1'
    subroutine clash1() bind(c)
    end subroutine
  end interface
  call clash1
end

! An EXTERNAL statement in a subprogram.
subroutine s2
  use m
  !ERROR: Two entities have the same global name 'clash2'
  external clash2
  call clash2
end

! Procedure declaration statements in a subprogram.
subroutine s3
  use m
  abstract interface
    subroutine iface() bind(c)
    end subroutine
  end interface
  !ERROR: Two entities have the same global name 'clash3'
  procedure(iface), bind(c) :: clash3
  !ERROR: Two entities have the same global name 'clash4'
  procedure(iface), bind(c, name='clash4') :: notclash4
  !ERROR: Two entities have the same global name 'clash5'
  procedure(real) :: clash5
  call clash3
  call notclash4
  print *, clash5()
end

! No conflicts: dummy procedures, abstract interfaces, generic interfaces,
! procedure pointers, and internal procedures are not global entities.
subroutine s4(ok1)
  use m
  external ok1
  abstract interface
    subroutine ok2() bind(c)
    end subroutine
  end interface
  interface ok3
    subroutine ok3specific()
    end subroutine
  end interface
  procedure(ok2), pointer :: ok4 => null()
  call ok1
 contains
  subroutine ok5
  end subroutine
end

! No conflict: an external procedure may be declared in more than one scope,
! and may be declared where it is also defined.
subroutine s5
  external twice
  call twice
end
subroutine s6
  interface
    subroutine twice()
    end subroutine
  end interface
  call twice
end
subroutine twice
end

! A common block is a global entity too (19.2 p1), whether the external
! procedure is declared in another scope or in the same one.
subroutine s7
  common /clash7/ x
end
subroutine s8
  !ERROR: Two entities have the same global name 'clash7'
  external clash7
  call clash7
end
subroutine s9
  !ERROR: Two entities have the same global name 'clash8'
  common /clash8/ x
  interface
    subroutine clash8()
    end subroutine
  end interface
  call s8(clash8)
end

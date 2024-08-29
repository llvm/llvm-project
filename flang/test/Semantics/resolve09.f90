! RUN: %python %S/test_errors.py %s %flang_fc1
integer :: y
procedure() :: a
procedure(real) :: b
call a  ! OK - can be function or subroutine
!ERROR: Cannot call subroutine 'a' like a function
c = a()
!ERROR: Cannot call function 'b' like a subroutine
call b
!ERROR: Cannot call function 'y' like a subroutine
call y
call x
!ERROR: Cannot call subroutine 'x' like a function
z = x()
end

subroutine s
  !ERROR: Cannot call function 'f' like a subroutine
  call f
  !ERROR: Cannot call subroutine 's' like a function
  i = s()
contains
  function f()
  end
end

subroutine s2
  ! subroutine vs. function is determined by use
  external :: a, b
  call a()
  !ERROR: Cannot call subroutine 'a' like a function
  x = a()
  x = b()
  !ERROR: Cannot call function 'b' like a subroutine
  call b()
end

subroutine s3
  ! subroutine vs. function is determined by use, even in internal subprograms
  external :: a
  procedure() :: b
contains
  subroutine s3a()
    x = a()
    call b()
  end
  subroutine s3b()
    !ERROR: Cannot call function 'a' like a subroutine
    call a()
    !ERROR: Cannot call subroutine 'b' like a function
    x = b()
  end
end

module m1
  !Function vs subroutine in a module is resolved to a subroutine if
  !no other information.
  external :: exts, extf, extunk
  procedure() :: procs, procf, procunk
contains
  subroutine s
    call exts()
    call procs()
    x = extf()
    x = procf()
  end
end

module m2
  use m1
 contains
  subroutine test
    call exts() ! ok
    call procs() ! ok
    call extunk() ! ok
    call procunk() ! ok
    x = extf() ! ok
    x = procf() ! ok
    !ERROR: Cannot call subroutine 'extunk' like a function
    !ERROR: Function result characteristics are not known
    x = extunk()
    !ERROR: Cannot call subroutine 'procunk' like a function
    !ERROR: Function result characteristics are not known
    x = procunk()
  end
end

module modulename
end

! Call to entity in global scope, even with IMPORT, NONE
subroutine s4
  block
    import, none
    integer :: i
    !ERROR: 'modulename' is not a callable procedure
    call modulename()
  end block
end

! Call to entity in global scope, even with IMPORT, NONE
subroutine s5
  block
    import, none
    integer :: i
    i = foo()
    !ERROR: Cannot call function 'foo' like a subroutine
    call foo()
  end block
end

subroutine s6
  call a6()
end
!ERROR: 'a6' was previously called as a subroutine
function a6()
  a6 = 0.0
end

subroutine s7
  x = a7()
end
!ERROR: 'a7' was previously called as a function
subroutine a7()
end

!OK: use of a8 and b8 is consistent
subroutine s8
  call a8()
  x = b8()
end
subroutine a8()
end
function b8()
  b8 = 0.0
end

subroutine s9
  type t
    procedure(), nopass, pointer :: p1, p2
  end type
  type(t) x
  print *, x%p1()
  call x%p2
  !ERROR: Cannot call function 'p1' like a subroutine
  call x%p1
  !ERROR: Cannot call subroutine 'p2' like a function
  print *, x%p2()
end subroutine

subroutine s10
  call a10
  !ERROR: Actual argument for 'a=' may not be a procedure
  print *, abs(a10)
end

subroutine s11
  real, pointer :: p(:)
  !ERROR: A NULL() pointer is not allowed for 'a=' intrinsic argument
  print *, rank(null())
  print *, rank(null(mold=p)) ! ok
end

! RUN: %python %S/test_errors.py %s %flang_fc1
! Other checks for declarations in PURE procedures
module m
  type t0
  end type
  type t1
   contains
    final :: final
  end type
  type t2
    type(t1), allocatable :: c
  end type
  type t3
    class(t1), allocatable :: c
  end type
  type t4
    class(t0), allocatable :: c
  end type
 contains
  impure subroutine final(x)
    type(t1) x
  end
  pure subroutine test
    !ERROR: 'x0' may not be a local variable in a pure subprogram
    !BECAUSE: 'x0' is polymorphic in a pure subprogram
    class(t0), allocatable :: x0
    !ERROR: 'x1' may not be a local variable in a pure subprogram
    !BECAUSE: 'x1' has an impure FINAL procedure 'final'
    type(t1) x1
    !WARNING: 'x1a' of derived type 't1' does not have a FINAL subroutine for its rank (1)
    type(t1), allocatable :: x1a(:)
    type(t1), parameter :: namedConst = t1() ! ok
    !ERROR: 'x2' may not be a local variable in a pure subprogram
    !BECAUSE: 'x2' has an impure FINAL procedure 'final'
    type(t2) x2
    !ERROR: 'x3' may not be a local variable in a pure subprogram
    !BECAUSE: 'x3' has an impure FINAL procedure 'final'
    type(t3) x3
    !ERROR: 'x4' may not be a local variable in a pure subprogram
    !BECAUSE: 'x4' has polymorphic component '%c' in a pure subprogram
    type(t4) x4
  end
end

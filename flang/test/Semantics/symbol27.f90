! RUN: %python %S/test_symbols.py %s %flang_fc1
!DEF: /m1a Module
module m1a
 !DEF: /m1a/foo PUBLIC DerivedType
 type :: foo
  !DEF: /m1a/foo/j ObjectEntity INTEGER(4)
  integer :: j
 end type
end module
!DEF: /m1b Module
module m1b
 !DEF: /m1b/foo PUBLIC (Function) Generic
 interface foo
  !DEF: /m1b/bar PUBLIC (Function) Subprogram REAL(4)
  module procedure :: bar
 end interface
contains
 !REF: /m1b/bar
 function bar()
 end function
end module
!DEF: /test1a (Subroutine) Subprogram
subroutine test1a
 !REF: /m1a
 use :: m1a
 !REF: /m1b
 use :: m1b
 !DEF: /test1a/foo (Function) Generic
 !DEF: /test1a/x ObjectEntity TYPE(foo)
 type(foo) :: x
 !DEF: /test1a/foo Use
 !REF: /m1b/bar
 print *, foo(1), foo()
end subroutine
!DEF: /test1b (Subroutine) Subprogram
subroutine test1b
 !REF: /m1b
 use :: m1b
 !REF: /m1a
 use :: m1a
 !DEF: /test1b/foo (Function) Generic
 !DEF: /test1b/x ObjectEntity TYPE(foo)
 type(foo) :: x
 !DEF: /test1b/foo Use
 !REF: /m1b/bar
 print *, foo(1), foo()
end subroutine

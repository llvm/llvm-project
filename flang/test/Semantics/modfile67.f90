!RUN: %flang_fc1 -fsyntax-only -J%S/Inputs %s

#if 0
!modfile67.mod was produced from this source, and must be read into this
!compilation from its module file in order to truly test this fix.
module modfile67
  type t
    procedure(foo), nopass, pointer :: p
  end type
 contains
  pure function foo(n,a) result(r)
    integer, intent(in) :: n
    real, intent(in), dimension(n) :: a
    logical, dimension(size(a)) :: r
    r = .false.
  end
  type(t) function fooptr(f)
    procedure(foo) f
    fooptr%p => f
  end
end
#endif

program test
  use modfile67
  type(t) x
  x = fooptr(bar) ! ensure no bogus error about procedure incompatibility
 contains
  pure function bar(n,a) result(r)
    integer, intent(in) :: n
    real, intent(in), dimension(n) :: a
    logical, dimension(size(a)) :: r
    r = .false.
  end
end

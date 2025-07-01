! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic -Werror
! F'2023 C7108 is portably unenforced.
module m
  type foo
    integer n
  end type
  interface foo
    procedure bar0, bar1, bar2, bar3
  end interface
 contains
  type(foo) function bar0(n)
    integer, intent(in) :: n
    print *, 'bar0'
    bar0%n = n
  end
  type(foo) function bar1()
    print *, 'bar1'
    bar1%n = 1
  end
  type(foo) function bar2(a)
    real, intent(in) :: a
    print *, 'bar2'
    bar2%n = a
  end
  type(foo) function bar3(L)
    logical, intent(in) :: L
    print *, 'bar3'
    bar3%n = merge(4,5,L)
  end
end

program p
  use m
  type(foo) x
  x = foo(); print *, x       ! ok, not ambiguous
  !PORTABILITY: Reference to generic function 'foo' (resolving to specific 'bar0') is ambiguous with a structure constructor of the same name [-Wambiguous-structure-constructor]
  x = foo(2); print *, x      ! ambigous
  !PORTABILITY: Reference to generic function 'foo' (resolving to specific 'bar2') is ambiguous with a structure constructor of the same name [-Wambiguous-structure-constructor]
  x = foo(3.); print *, x     ! ambiguous due to data conversion
  x = foo(.true.); print *, x ! ok, not ambigous
end

! RUN: %python %S/test_errors.py %s %flang_fc1
subroutine test1
  !ERROR: Generic interface 'foo' has both a function and a subroutine
  interface foo
    subroutine s1(x)
    end subroutine
    subroutine s2(x, y)
    end subroutine
    function f()
    end function
  end interface
end subroutine

subroutine test2
  !ERROR: Generic interface 'foo' has both a function and a subroutine
  interface foo
    function f1(x)
    end function
    subroutine s()
    end subroutine
    function f2(x, y)
    end function
  end interface
end subroutine

module test3
  !ERROR: Generic interface 'foo' has both a function and a subroutine
  interface foo
    module procedure s
    module procedure f
  end interface
contains
  subroutine s(x)
  end subroutine
  function f()
  end function
end module

subroutine test4
  type foo
  end type
  !ERROR: Generic interface 'foo' may only contain functions due to derived type with same name
  interface foo
    subroutine s()
    end subroutine
  end interface
end subroutine

subroutine test5
  interface foo
    function f1()
    end function
  end interface
  interface bar
    subroutine s1()
    end subroutine
    subroutine s2(x)
    end subroutine
  end interface
  !ERROR: Cannot call function 'foo' like a subroutine
  call foo()
  !ERROR: Cannot call subroutine 'bar' like a function
  x = bar()
end subroutine

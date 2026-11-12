! RUN: %python %S/test_errors.py %s %flang_fc1
! Ensures that a generic's shadowed procedure or derived type
! can be overridden by a valid interior interface definition
! in some cases.

module m1
 contains
  subroutine foo
  end subroutine
  subroutine test
    interface foo
      subroutine foo(n)
        integer, intent(in) :: n
      end subroutine
    end interface
    call foo(1)
  end subroutine
end module

module m2
 contains
  subroutine test
    interface foo
      subroutine foo(n)
        integer, intent(in) :: n
      end subroutine
    end interface
    call foo(1)
  end subroutine
  subroutine foo
  end subroutine
end module

module m3
  interface
    subroutine foo
    end subroutine
  end interface
 contains
  subroutine test
    interface foo
      subroutine foo(n)
        integer, intent(in) :: n
      end subroutine
    end interface
    call foo(1)
  end subroutine
end module

module m4a
 contains
  subroutine foo
  end subroutine
end module
module m4b
  use m4a
 contains
  subroutine test
    interface foo
      subroutine foo(n)
        integer, intent(in) :: n
      end subroutine
    end interface
    call foo(1)
  end subroutine
end module

module m5
  type bar
  end type
 contains
  subroutine test
    interface bar
      real function bar()
      end function
    end interface
    print *, bar()
  end subroutine
end module

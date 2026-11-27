!RUN: %python %S/test_errors.py %s %flang_fc1 -Werror -pedantic
module m
  interface
    integer function foo1()
    end function
    integer function foo2(j)
    end function
    integer function foo3()
    end function
  end interface
end module

subroutine test()
  integer, external :: foo1
!WARNING: The external interface 'foo2' is not compatible with an earlier definition (distinct numbers of dummy arguments) [-Wexternal-interface-mismatch]
  integer, external :: foo2
  integer, external :: foo3
  call bar(foo1())
  call bar(foo2())
  call baz(foo3)
end subroutine

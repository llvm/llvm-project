! RUN: %python %S/test_errors.py %s %flang_fc1
! C722 A function name shall not be declared with an asterisk type-param-value 
! unless it is of type CHARACTER and is the name of a dummy function or the 
! name of the result of an external function.
subroutine s()

  type derived(param)
    integer, len :: param
  end type
  type(derived(34)) :: a

  procedure(character(len=*)) :: externCharFunc
  !ERROR: An assumed (*) type parameter may be used only for a (non-statement function) dummy argument, associate name, named constant, or external function result
  procedure(type(derived(param =*))) :: externDerivedFunc

  interface
    subroutine subr(dummyFunc)
      character(len=*) :: dummyFunc
    end subroutine subr
  end interface

  contains
    function works()
      type(derived(param=4)) :: works
    end function works

  !ERROR: An assumed (*) type parameter may be used only for a (non-statement function) dummy argument, associate name, named constant, or external function result
    function fails1()
      character(len=*) :: fails1
    end function fails1

  !ERROR: An assumed (*) type parameter may be used only for a (non-statement function) dummy argument, associate name, named constant, or external function result
    function fails2()
  !ERROR: An assumed (*) type parameter may be used only for a (non-statement function) dummy argument, associate name, named constant, or external function result
      type(derived(param=*)) :: fails2
    end function fails2

end subroutine s

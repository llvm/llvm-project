! RUN: %S/test_errors.sh %s %t %f18
! C721 A type-param-value of * shall be used only
! * to declare a dummy argument,
! * to declare a named constant,
! * in the type-spec of an ALLOCATE statement wherein each allocate-object is 
!   a dummy argument of type CHARACTER with an assumed character length,
! * in the type-spec or derived-type-spec of a type guard statement (11.1.11), 
!   or
! * in an external function, to declare the character length parameter of the function result.
subroutine s(arg)
  character(len=*), pointer :: arg
  character*(*), parameter  :: cvar1 = "abc"
  character*4,  cvar2
  character(len=4_4) :: cvar3
  !ERROR: An assumed (*) type parameter may be used only for a (non-statement function) dummy argument, associate name, named constant, or external function result
  character(len=*) :: cvar4

  type derived(param)
    integer, len :: param
    class(*), allocatable :: x
  end type
  type(derived(34)) :: a
  interface
    function fun()
      character(len=4) :: fun
    end function fun
  end interface

  select type (ax => a%x)
    type is (integer)
      print *, "hello"
    type is (character(len=*))
      print *, "hello"
    class is (derived(param=*))
      print *, "hello"
    class default
      print *, "hello"
  end select

  allocate (character(len=*) :: arg)
end subroutine s

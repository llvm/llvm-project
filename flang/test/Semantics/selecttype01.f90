! RUN: %S/test_errors.sh %s %t %f18
! Test for checking select type constraints,
module m1
  use ISO_C_BINDING
  type shape
    integer :: color
    logical :: filled
    integer :: x
    integer :: y
  end type shape

  type, extends(shape) :: rectangle
    integer :: length
    integer :: width
  end type rectangle

  type, extends(rectangle) :: square
  end type square

  type, extends(square) :: extsquare
  end type

  type :: unrelated
    logical :: some_logical
  end type

  type withSequence
    SEQUENCE
    integer :: x
  end type

  type, BIND(C) :: withBind
    INTEGER(c_int) ::int_in_c
  end type

  TYPE(shape), TARGET :: shape_obj
  TYPE(rectangle), TARGET :: rect_obj
  TYPE(square), TARGET :: squr_obj
  !define polymorphic objects
  class(*), pointer :: unlim_polymorphic
  class(shape), pointer :: shape_lim_polymorphic
end
module m
  type :: t(n)
    integer, len :: n
  end type
contains
  subroutine CheckC1160( a )
    class(*), intent(in) :: a
    select type ( a )
      !ERROR: The type specification statement must have LEN type parameter as assumed
      type is ( character(len=10) ) !<-- assumed length-type
      ! OK
      type is ( character(len=*) )
      !ERROR: The type specification statement must have LEN type parameter as assumed
      type is ( t(n=10) )
      ! OK
      type is ( t(n=*) )   !<-- assumed length-type
      !ERROR: Derived type 'character' not found
      class is ( character(len=10) ) !<-- assumed length-type
    end select
  end subroutine

  subroutine s()
    type derived(param)
      integer, len :: param
      class(*), allocatable :: x
    end type
    TYPE(derived(10)) :: a
    select type (ax => a%x)
      class is (derived(param=*))
        print *, "hello"
    end select
  end subroutine s
end module

subroutine CheckC1157
  use m1
  integer, parameter :: const_var=10
  !ERROR: Selector is not a named variable: 'associate-name =>' is required
  select type(10)
  end select
  !ERROR: Selector is not a named variable: 'associate-name =>' is required
  select type(const_var)
  end select
  !ERROR: Selector is not a named variable: 'associate-name =>' is required
  select type (4.999)
  end select
  !ERROR: Selector is not a named variable: 'associate-name =>' is required
  select type (shape_obj%x)
  end select
end subroutine

!CheckPloymorphicSelectorType
subroutine CheckC1159a
  integer :: int_variable
  real :: real_variable
  complex :: complex_var = cmplx(3.0, 4.0)
  logical :: log_variable
  character (len=10) :: char_variable = "OM"
  !ERROR: Selector 'int_variable' in SELECT TYPE statement must be polymorphic
  select type (int_variable)
  end select
  !ERROR: Selector 'real_variable' in SELECT TYPE statement must be polymorphic
  select type (real_variable)
  end select
  !ERROR: Selector 'complex_var' in SELECT TYPE statement must be polymorphic
  select type(complex_var)
  end select
  !ERROR: Selector 'logical_variable' in SELECT TYPE statement must be polymorphic
  select type(logical_variable)
  end select
  !ERROR: Selector 'char_variable' in SELECT TYPE statement must be polymorphic
  select type(char_variable)
  end select
end

subroutine CheckC1159b
  integer :: x
  !ERROR: Selector 'x' in SELECT TYPE statement must be polymorphic
  select type (a => x)
  !ERROR: If selector is not unlimited polymorphic, an intrinsic type specification must not be specified in the type guard statement
  type is (integer)
    print *,'integer ',a
  end select
end

subroutine CheckC1159c
  !ERROR: Selector 'x' in SELECT TYPE statement must be polymorphic
  select type (a => x)
  !ERROR: If selector is not unlimited polymorphic, an intrinsic type specification must not be specified in the type guard statement
  type is (integer)
    print *,'integer ',a
  end select
end

subroutine s(arg)
  class(*) :: arg
    select type (arg)
        type is (integer)
    end select
end

subroutine CheckC1161
  use m1
  shape_lim_polymorphic => rect_obj
  select type(shape_lim_polymorphic)
    !ERROR: The type specification statement must not specify a type with a SEQUENCE attribute or a BIND attribute
    type is (withSequence)
    !ERROR: The type specification statement must not specify a type with a SEQUENCE attribute or a BIND attribute
    type is (withBind)
  end select
end

subroutine CheckC1162
  use m1
  class(rectangle), pointer :: rectangle_polymorphic
  !not unlimited polymorphic objects
  select type (rectangle_polymorphic)
    !ERROR: Type specification 'shape' must be an extension of TYPE 'rectangle'
    type is (shape)
    !ERROR: Type specification 'unrelated' must be an extension of TYPE 'rectangle'
    type is (unrelated)
    !all are ok
    type is (square)
    type is (extsquare)
    !Handle same types
    type is (rectangle)
    !ERROR: If selector is not unlimited polymorphic, an intrinsic type specification must not be specified in the type guard statement
    type is(integer)
    !ERROR: If selector is not unlimited polymorphic, an intrinsic type specification must not be specified in the type guard statement
    type is(real)
    !ERROR: If selector is not unlimited polymorphic, an intrinsic type specification must not be specified in the type guard statement
    type is(logical)
    !ERROR: If selector is not unlimited polymorphic, an intrinsic type specification must not be specified in the type guard statement
    type is(character(len=*))
    !ERROR: If selector is not unlimited polymorphic, an intrinsic type specification must not be specified in the type guard statement
    type is(complex)
  end select

  !Unlimited polymorphic objects are allowed.
  unlim_polymorphic => rect_obj
  select type (unlim_polymorphic)
    type is (shape)
    type is (unrelated)
  end select
end

subroutine CheckC1163
  use m1
  !assign dynamically
  shape_lim_polymorphic => rect_obj
  unlim_polymorphic => shape_obj
  select type (shape_lim_polymorphic)
    type is (shape)
    !ERROR: Type specification 'shape' conflicts with previous type specification
    type is (shape)
    class is (square)
    !ERROR: Type specification 'square' conflicts with previous type specification
    class is (square)
  end select
  select type (unlim_polymorphic)
    type is (INTEGER(4))
    type is (shape)
    !ERROR: Type specification 'INTEGER(4)' conflicts with previous type specification
    type is (INTEGER(4))
  end select
end

subroutine CheckC1164
  use m1
  shape_lim_polymorphic => rect_obj
  unlim_polymorphic => shape_obj
  select type (shape_lim_polymorphic)
    CLASS DEFAULT
    !ERROR: Type specification 'DEFAULT' conflicts with previous type specification
    CLASS DEFAULT
    TYPE IS (shape)
    TYPE IS (rectangle)
    !ERROR: Type specification 'DEFAULT' conflicts with previous type specification
    CLASS DEFAULT
  end select

  !Saving computation if some error in guard by not computing RepeatingCases
  select type (shape_lim_polymorphic)
    CLASS DEFAULT
    CLASS DEFAULT
    !ERROR: The type specification statement must not specify a type with a SEQUENCE attribute or a BIND attribute
    TYPE IS(withSequence)
  end select
end subroutine

subroutine WorkingPolymorphism
  use m1
  !assign dynamically
  shape_lim_polymorphic => rect_obj
  unlim_polymorphic => shape_obj
  select type (shape_lim_polymorphic)
    type is  (shape)
      print *, "hello shape"
    type is  (rectangle)
      print *, "hello rect"
    type is  (square)
      print *, "hello square"
    CLASS DEFAULT
      print *, "default"
  end select
  print *, "unlim polymorphism"
  select type (unlim_polymorphic)
    type is  (shape)
      print *, "hello shape"
    type is  (rectangle)
      print *, "hello rect"
    type is  (square)
      print *, "hello square"
    CLASS DEFAULT
      print *, "default"
  end select
end

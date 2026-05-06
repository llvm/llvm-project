! RUN: %flang_fc1 -fsyntax-only %s

module associated_func_call
  implicit none
  private
  public :: type_t
  public :: test_function_i
  abstract interface
    function test_function_i() result(passes)
      implicit none
      logical passes
    end function
  end interface

  type type_t
    private
    procedure(test_function_i), pointer, nopass :: test_function_ => null()
  contains
    generic :: operator(==) => equals
    procedure, private :: equals
  end type

  interface type_t
    module function construct(test_function) result(test_description)
      implicit none
      procedure(test_function_i), intent(in), pointer :: test_function
      type(type_t) test_description
    end function
  end interface

  interface
    elemental module function equals(lhs, rhs) result(lhs_eq_rhs)
      implicit none
      class(type_t), intent(in) :: lhs, rhs
      logical lhs_eq_rhs
    end function
  end interface
  
contains
    module procedure construct
      test_description%test_function_ => test_function
    end procedure
    module procedure equals
      lhs_eq_rhs = associated(lhs%test_function_, rhs%test_function_)
    end procedure
end module associated_func_call

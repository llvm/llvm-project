! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic

module m
  type, bind(c) :: explicit_bind_c
    real a
  end type
  type :: interoperable1
    type(explicit_bind_c) a
  end type
  type, extends(interoperable1) :: interoperable2
    real b
  end type
  type :: non_interoperable1
    real, allocatable :: a
  end type
  type :: non_interoperable2
    type(non_interoperable1) b
  end type
  type :: no_bind_c
    real a
  end type
  type, bind(c) :: has_bind_c
    !WARNING: Derived type of component 'a' of an interoperable derived type should have the BIND attribute
    type(no_bind_c) :: a
  end type
  interface
    subroutine sub_bind_c_1(x_bind_c) bind(c)
      import explicit_bind_c
      type(explicit_bind_c), intent(in) :: x_bind_c
    end
    subroutine sub_bind_c_2(x_interop1) bind(c)
      import interoperable1
      !WARNING: The derived type of an interoperable object should be BIND(C)
      type(interoperable1), intent(in) :: x_interop1
    end
    subroutine sub_bind_c_3(x_interop2) bind(c)
      import interoperable2
      !WARNING: The derived type of an interoperable object should be BIND(C)
      type(interoperable2), intent(in) :: x_interop2
    end
    subroutine sub_bind_c_4(x_non_interop1) bind(c)
      import non_interoperable1
      !ERROR: The derived type of an interoperable object must be interoperable, but is not
      type(non_interoperable1), intent(in) :: x_non_interop1
    end
    subroutine sub_bind_c_5(x_non_interop2) bind(c)
      import non_interoperable2
      !ERROR: The derived type of an interoperable object must be interoperable, but is not
      type(non_interoperable2), intent(in) :: x_non_interop2
    end
  end interface
end

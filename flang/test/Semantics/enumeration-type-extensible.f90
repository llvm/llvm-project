! RUN: %python %S/test_errors.py %s %flang_fc1
! An enumeration type is not a derived type (F2023 7.6.2), so it is not an
! extensible type (F2023 7.5.7.1).

module enum_ext_mod
  enumeration type :: color
    enumerator :: red, green
  end enumeration type

  type :: base
  end type

  !ERROR: The parent type is not extensible
  type, extends(color) :: child
    integer :: n = 0
  end type

  ! The rejected EXTENDS must not leave a hidden parent component behind.
  type, extends(child) :: grandchild
    integer :: k
  end type

  !ERROR: Non-extensible derived type 'color' may not be used with CLASS keyword
  class(color), allocatable :: x

contains
  subroutine use_child()
    type(child) :: c
    type(grandchild) :: g
    c = child(n=1)
    c%n = c%n + 1
    g = grandchild(k=3)
  end subroutine

  subroutine classof_enum(e)
    type(color), intent(in) :: e
    !ERROR: CLASSOF requires a data-ref of extensible type
    classof(e), allocatable :: f
  end subroutine

  !ERROR: Non-extensible derived type 'color' may not be used with CLASS keyword
  class(color) function class_result()
    allocatable :: class_result
  end function

  subroutine type_is_ok(u)
    class(*), intent(in) :: u
    select type (u)
    type is (color)
    end select
  end subroutine

  subroutine class_is_enum(u)
    class(*), intent(in) :: u
    select type (u)
    !ERROR: An enumeration type may not appear in a CLASS IS type guard statement
    class is (color)
    end select
  end subroutine

  subroutine type_is_not_unlimited(b)
    class(base), intent(in) :: b
    select type (b)
    !ERROR: If selector is not unlimited polymorphic, an enumeration type specification must not be specified in the type guard statement
    type is (color)
    end select
  end subroutine

  subroutine pointer_to_unlimited(q)
    class(*), pointer, intent(in) :: q
    type(color), pointer :: p
    !ERROR: Pointer type must be unlimited polymorphic or non-extensible derived type when target is unlimited polymorphic
    p => q
  end subroutine

  subroutine same_type(a, b)
    type(color), intent(in) :: a, b
    !ERROR: Actual argument for 'a=' has type 'color', but was expected to be an extensible or unlimited polymorphic type
    print *, same_type_as(a, b)
  end subroutine
end module

module enum_ext_implicit_mod
  use enum_ext_mod, only: color
  !ERROR: Non-extensible derived type 'color' may not be used with CLASS keyword
  implicit class(color) (c)
end module

module enum_ext_dio_mod
  enumeration type :: color
    enumerator :: red, green
  end enumeration type
  interface write(formatted)
    subroutine wf(dtv, unit, iotype, vlist, iostat, iomsg)
      import color
      !ERROR: Dummy argument 'dtv' of a defined input/output procedure must have a derived type
      type(color), intent(in) :: dtv
      integer, intent(in) :: unit
      character(*), intent(in) :: iotype
      integer, intent(in) :: vlist(:)
      integer, intent(out) :: iostat
      character(*), intent(inout) :: iomsg
    end subroutine
  end interface
end module

! RUN: %S/test_errors.sh %s %t %f18
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

  TYPE(shape), TARGET :: shape_obj
  TYPE(rectangle), TARGET :: rect_obj
 !define polymorphic objects
  class(shape), pointer :: shape_lim_polymorphic
end
subroutine C1165a
  use m1
  shape_lim_polymorphic => rect_obj
  label : select type (shape_lim_polymorphic)
  end select label
  label1 : select type (shape_lim_polymorphic)
  !ERROR: SELECT TYPE construct name required but missing
  end select
  select type (shape_lim_polymorphic)
  !ERROR: SELECT TYPE construct name unexpected
  end select label2
  select type (shape_lim_polymorphic)
  end select
end subroutine
subroutine C1165b
  use m1
  shape_lim_polymorphic => rect_obj
!type-guard-stmt realted checks
label : select type (shape_lim_polymorphic)
  type is (shape) label
  end select label
 select type (shape_lim_polymorphic)
  !ERROR: SELECT TYPE name not allowed
  type is (shape) label
  end select
label : select type (shape_lim_polymorphic)
  !ERROR: SELECT TYPE name mismatch
  type is (shape) labelll
  end select label
end subroutine

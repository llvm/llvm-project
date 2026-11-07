! RUN: %python %S/test_errors.py %s %flang_fc1

! These ordinary Fortran specifics are distinguishable by the type of x.
! Both match a bare NULL(), however, and all their matching distances are
! zero. Removing the nonfirst passed object must leave comparable effective
! argument vectors, so the ambiguous calls are diagnosed instead of aborting.
module mixed_null_bindings
  type :: t
  contains
    procedure, pass(this) :: with_object
    procedure, nopass :: without_object
    generic :: choose => with_object, without_object
    generic :: choose_reversed => without_object, with_object
    generic :: integer_only => with_object
    generic :: real_only => without_object
  end type
contains
  subroutine with_object(x, this)
    integer, pointer, intent(in) :: x
    class(t), intent(in) :: this
  end subroutine
  subroutine without_object(x)
    real, pointer, intent(in) :: x
  end subroutine
  subroutine caller(obj, ip, rp)
    type(t) :: obj
    integer, pointer :: ip
    real, pointer :: rp
    ! F2023 16.9.155p6 requires MOLD= when type information is needed to
    ! resolve a generic reference. Check both binding orders and a keyword.
    !ERROR: The actual arguments to the generic procedure 'choose' matched multiple specific procedures, perhaps due to use of NULL() without MOLD= or an actual procedure with an implicit interface
    call obj%choose(null())
    !ERROR: The actual arguments to the generic procedure 'choose_reversed' matched multiple specific procedures, perhaps due to use of NULL() without MOLD= or an actual procedure with an implicit interface
    call obj%choose_reversed(x=null())
    ! MOLD= makes each call resolvable, selecting either PASS or NOPASS.
    call obj%choose(null(mold=ip))
    call obj%choose(null(mold=rp))
    call obj%choose_reversed(x=null(mold=ip))
    call obj%choose_reversed(x=null(mold=rp))
    ! A singleton generic needs no type information to select its specific.
    call obj%integer_only(null())
    call obj%real_only(null())
  end subroutine
end module

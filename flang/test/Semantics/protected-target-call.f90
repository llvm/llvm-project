! RUN: %python %S/test_errors.py %s %flang_fc1
! Actual arguments that are PROTECTED_TARGET pointers or subobjects of their
! targets (F'2028 C872, C873, C874).

module m
  type :: t
    integer :: n
    integer, pointer :: next
   contains
    procedure :: in_method, method
  end type
  interface gen
    module procedure gen_int, gen_real
  end interface
 contains
  subroutine in_method(this)
    class(t), intent(in) :: this
  end
  subroutine method(this)
    class(t) :: this
  end
  subroutine ptr_pt(d)
    integer, pointer, protected_target :: d
  end
  subroutine ptr_pt_out(d)
    integer, pointer, protected_target, intent(out) :: d
  end
  subroutine ptr_pt_in_array(d)
    integer, pointer, protected_target, intent(in) :: d(:)
  end
  subroutine ptr(d)
    integer, pointer :: d
  end
  subroutine ptr_in(d)
    integer, pointer, intent(in) :: d
  end
  subroutine ptr_inout(d)
    integer, pointer, intent(in out) :: d
  end
  subroutine ptr_in_array(d)
    integer, pointer, intent(in) :: d(:)
  end
  subroutine obj_in(d)
    integer, intent(in) :: d
  end
  subroutine obj(d)
    integer :: d
  end
  subroutine obj_value(d)
    integer, value :: d
  end
  subroutine obj_out(d)
    integer, intent(out) :: d
  end
  subroutine obj_inout(d)
    integer, intent(in out) :: d
  end
  subroutine gen_int(d)
    integer :: d
  end
  subroutine gen_real(d)
    real :: d
  end
  subroutine ar_ptr(d)
    integer, pointer :: d(..)
  end
  subroutine ar_ptr_pt(d)
    integer, pointer, protected_target :: d(..)
  end
  subroutine ar_obj(d)
    integer :: d(..)
  end
  subroutine ar_obj_in(d)
    integer, intent(in) :: d(..)
  end
  function get() result(r)
    integer, pointer, protected_target :: r
    allocate(r, source=1)
  end
end

subroutine calls(p, pa, pt, w)
  use m
  use iso_c_binding, only: c_ptr, c_f_pointer
  integer, pointer, protected_target :: p, pa(:)
  type(t), pointer, protected_target :: pt
  integer, pointer :: w
  real, pointer, protected_target :: pr
  type(c_ptr) :: cp
  procedure(ptr_pt), pointer :: proc_pt
  procedure(ptr), pointer :: proc
  call ptr_pt(p)
  call ptr_pt_out(p)
  call ptr_pt_in_array(pa(1:2))
  call ptr_pt(w)
  !ERROR: The target of PROTECTED_TARGET pointer 'p' may not be associated with POINTER dummy argument 'd=', which does not have the PROTECTED_TARGET attribute
  call ptr(p)
  !ERROR: The target of PROTECTED_TARGET pointer 'p' may not be associated with POINTER dummy argument 'd=', which does not have the PROTECTED_TARGET attribute
  call ptr_in(p)
  !ERROR: The target of PROTECTED_TARGET pointer 'pa' may not be associated with POINTER dummy argument 'd=', which does not have the PROTECTED_TARGET attribute
  call ptr_in_array(pa(1:2))
  !ERROR: The target of PROTECTED_TARGET pointer 'r' may not be associated with POINTER dummy argument 'd=', which does not have the PROTECTED_TARGET attribute
  call ptr_in(get())
  call obj_in(p)
  call obj_in(pa(1))
  call obj_in(pt%n)
  call obj_in(get())
  call obj((p))
  !ERROR: The target of PROTECTED_TARGET pointer 'p' may not be associated with dummy argument 'd=', which does not have the INTENT(IN) attribute
  call obj(p)
  !ERROR: The target of PROTECTED_TARGET pointer 'pa' may not be associated with dummy argument 'd=', which does not have the INTENT(IN) attribute
  call obj(pa(2))
  !ERROR: The target of PROTECTED_TARGET pointer 'pt' may not be associated with dummy argument 'd=', which does not have the INTENT(IN) attribute
  call obj(pt%n)
  !ERROR: The target of PROTECTED_TARGET pointer 'r' may not be associated with dummy argument 'd=', which does not have the INTENT(IN) attribute
  call obj(get())
  !ERROR: The target of PROTECTED_TARGET pointer 'p' may not be associated with dummy argument 'd=', which does not have the INTENT(IN) attribute
  call obj_value(p)
  !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'd=' is not definable
  !BECAUSE: 'p' has the PROTECTED_TARGET attribute
  call obj_out(p)
  !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'd=' is not definable
  !BECAUSE: 'pa' has the PROTECTED_TARGET attribute
  call obj_inout(pa(1))
  !ERROR: The target of PROTECTED_TARGET pointer 'p' may not be associated with dummy argument 'd=', which does not have the INTENT(IN) attribute
  call gen(p)
  ! A pointer component of the target is a subobject of the target when the
  ! dummy argument is a pointer, but its own target is not.
  call obj(pt%next)
  !ERROR: Pointer component 'next' of the target of PROTECTED_TARGET pointer 'pt' may not be associated with POINTER dummy argument 'd=', which does not have the PROTECTED_TARGET attribute
  call ptr(pt%next)
  !ERROR: Pointer component 'next' of the target of PROTECTED_TARGET pointer 'pt' may not be associated with POINTER dummy argument 'd=', which does not have the PROTECTED_TARGET attribute
  call ptr_in(pt%next)
  !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'd=' is not definable
  !BECAUSE: 'pt' has the PROTECTED_TARGET attribute
  call ptr_inout(pt%next)
  call ptr_pt(pt%next)
  call pt%in_method()
  !ERROR: The target of PROTECTED_TARGET pointer 'pt' may not be associated with dummy argument 'this=', which does not have the INTENT(IN) attribute
  call pt%method()
  proc_pt => ptr_pt
  call proc_pt(p)
  proc => ptr
  !ERROR: The target of PROTECTED_TARGET pointer 'p' may not be associated with POINTER dummy argument 'd=', which does not have the PROTECTED_TARGET attribute
  call proc(p)
  ! Intrinsic procedures
  print *, associated(p), associated(pa, pa), size(pa), lbound(pa), kind(p)
  print *, sum(pa), maxval(pa), loc(p)
  call c_f_pointer(cp, p)
  ! INTENT(OUT) arguments of intrinsic procedures are still subject to C866.
  !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'harvest=' is not definable
  !BECAUSE: 'pr' has the PROTECTED_TARGET attribute
  call random_number(pr)
end

subroutine associate_names(p, ar)
  use m
  integer, pointer, protected_target :: p, ar(..)
  external :: ext
  ! Accepted: neither an ASSOCIATE name (F'2028 11.1.3.3p1) nor a RANK(n)
  ! associate name (11.1.12.3p3) has the PROTECTED_TARGET attribute.
  associate (a => p)
    call obj(a)
    call ptr_in(a)
    call ext(a)
    !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'd=' is not definable
    !BECAUSE: 'p' has the PROTECTED_TARGET attribute
    call obj_out(a)
  end associate
  select rank (r => ar)
  rank (0)
    call obj_in(r)
    call ptr_pt(r)
    call obj(r)
    call ptr(r)
    call ptr_inout(r)
    call ext(r)
    !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'd=' is not definable
    !BECAUSE: 'ar' has the PROTECTED_TARGET attribute
    call obj_out(r)
  rank default
    ! A RANK DEFAULT associate name has exactly the attributes of its
    ! selector (F'2028 11.1.12.3p2).
    call ar_obj_in(r)
    call ar_ptr_pt(r)
    !ERROR: The target of PROTECTED_TARGET pointer 'ar' may not be associated with POINTER dummy argument 'd=', which does not have the PROTECTED_TARGET attribute
    call ar_ptr(r)
    !ERROR: The target of PROTECTED_TARGET pointer 'ar' may not be associated with dummy argument 'd=', which does not have the INTENT(IN) attribute
    call ar_obj(r)
    !ERROR: The target of PROTECTED_TARGET pointer 'ar' may not be an actual argument to a procedure with an implicit interface
    !ERROR: Assumed rank argument 'r' requires an explicit interface
    call ext(r)
  end select
end

subroutine implicit_interface(p, pa, dp)
  integer, pointer, protected_target :: p, pa(:)
  external :: ext, dp
  procedure(), pointer :: ip
  !ERROR: The target of PROTECTED_TARGET pointer 'p' may not be an actual argument to a procedure with an implicit interface
  call ext(p)
  !ERROR: The target of PROTECTED_TARGET pointer 'pa' may not be an actual argument to a procedure with an implicit interface
  call ext(pa(1))
  !ERROR: The target of PROTECTED_TARGET pointer 'pa' may not be an actual argument to a procedure with an implicit interface
  print *, ifun(pa)
  !ERROR: The target of PROTECTED_TARGET pointer 'p' may not be an actual argument to a procedure with an implicit interface
  call dp(p)
  !ERROR: The target of PROTECTED_TARGET pointer 'p' may not be an actual argument to a procedure with an implicit interface
  call ip(p)
  call ext((p))
  call ext(p + 1)
end

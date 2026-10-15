! RUN: %python %S/test_errors.py %s %flang_fc1
! F'2028 C872-C874: argument association must preserve target protection.

module protected_target_calls
  type t
    integer :: value
  contains
    procedure :: read_value
    procedure :: default_value
  end type
  type wrapper
    type(t), pointer, protected_target :: p
  end type
  interface pointer_generic
    module procedure pointer_array, elemental_read
  end interface
  interface default_generic
    module procedure default_array, elemental_read
  end interface
contains
  subroutine test(p, a, obj, container, ordinary)
    integer, pointer, protected_target :: p, a(:)
    type(t), pointer, protected_target :: obj
    type(wrapper) :: container
    integer, pointer :: ordinary
    external :: implicit_scalar, implicit_array
    call protected_pointer(p)
    call protected_pointer(ordinary)
    call protected_pointer_out(p)
    call protected_pointer_out(ordinary)
    !ERROR: A PROTECTED_TARGET pointer or its subobject may not be associated with dummy argument 'x=' without PROTECTED_TARGET
    call pointer_in(p)
    !ERROR: A PROTECTED_TARGET pointer or its subobject may not be associated with dummy argument 'x=' without PROTECTED_TARGET
    call pointer_in((.true. ? p : ordinary))
    !ERROR: A PROTECTED_TARGET pointer or its subobject may not be associated with dummy argument 'x=' without PROTECTED_TARGET
    call pointer_out(p)
    !ERROR: A PROTECTED_TARGET pointer or its subobject may not be associated with dummy argument 'x=' without PROTECTED_TARGET
    call pointer_in(a(1))
    !ERROR: A PROTECTED_TARGET pointer or its subobject may not be associated with dummy argument 'x=' without PROTECTED_TARGET
    call pointer_in(obj%value)
    !ERROR: A PROTECTED_TARGET pointer or its subobject may not be associated with dummy argument 'x=' without PROTECTED_TARGET
    call pointer_in(container%p%value)
    call value_in(p)
    call value_in(a(1))
    call value_in(obj%value)
    call value_in(container%p%value)
    !ERROR: A PROTECTED_TARGET pointer or its subobject may be associated with nonpointer dummy argument 'x=' only with INTENT(IN)
    call value_default(p)
    !ERROR: A PROTECTED_TARGET pointer or its subobject may be associated with nonpointer dummy argument 'x=' only with INTENT(IN)
    call value_default((.true. ? p : ordinary))
    !ERROR: A PROTECTED_TARGET pointer or its subobject may be associated with nonpointer dummy argument 'x=' only with INTENT(IN)
    call value_out(p)
    !ERROR: A PROTECTED_TARGET pointer or its subobject may be associated with nonpointer dummy argument 'x=' only with INTENT(IN)
    call value_inout(obj%value)
    !ERROR: A PROTECTED_TARGET pointer or its subobject may be associated with nonpointer dummy argument 'x=' only with INTENT(IN)
    call value_copy(p)
    call value_default((p))
    call value_default(p + 1)
    !ERROR: A PROTECTED_TARGET pointer or its subobject may not be an actual argument to a procedure with an implicit interface
    call implicit_scalar(p)
    !ERROR: A PROTECTED_TARGET pointer or its subobject may not be an actual argument to a procedure with an implicit interface
    call implicit_scalar(a(1))
    !ERROR: A PROTECTED_TARGET pointer or its subobject may not be an actual argument to a procedure with an implicit interface
    call implicit_scalar(container%p%value)
    !ERROR: A PROTECTED_TARGET pointer or its subobject may not be an actual argument to a procedure with an implicit interface
    call implicit_array(a(1:3))
    call implicit_scalar((p))
    call implicit_scalar(p + 1)
    call obj%read_value()
    !ERROR: A PROTECTED_TARGET pointer or its subobject may be associated with nonpointer dummy argument 'self=' only with INTENT(IN)
    call obj%default_value()
    ! These constraints must not discard a specific during generic resolution.
    !ERROR: A PROTECTED_TARGET pointer or its subobject may not be associated with dummy argument 'x=' without PROTECTED_TARGET
    call pointer_generic(a)
    !ERROR: A PROTECTED_TARGET pointer or its subobject may be associated with nonpointer dummy argument 'x=' only with INTENT(IN)
    call default_generic(a)
    associate (alias => container%p%value)
      call value_in(alias)
      call value_default(alias)
      call implicit_scalar(alias)
    end associate
    call value_in(result_pointer())
    !ERROR: A PROTECTED_TARGET pointer or its subobject may not be associated with dummy argument 'x=' without PROTECTED_TARGET
    call pointer_in(result_pointer())
    !ERROR: A PROTECTED_TARGET pointer or its subobject may be associated with nonpointer dummy argument 'x=' only with INTENT(IN)
    call value_default(result_pointer())
    !ERROR: A PROTECTED_TARGET pointer or its subobject may not be an actual argument to a procedure with an implicit interface
    call implicit_scalar(result_pointer())
    print *, associated(p), associated(p, ordinary), size(a), p
  end subroutine

  subroutine null_mold(p)
    integer, pointer, protected_target :: p, r
    integer, pointer :: q
    r => null(p)
    r => null(mold=p)
    q => null(p)
    call protected_pointer(null(mold=p))
  end subroutine

  subroutine intrinsic_output(p)
    use, intrinsic :: iso_c_binding, only: c_ptr, c_f_pointer
    real, pointer, protected_target :: p
    type(c_ptr) :: cp
    !ERROR: A PROTECTED_TARGET pointer or its subobject may be associated with nonpointer dummy argument 'harvest=' only with INTENT(IN)
    call random_number(p)
    !ERROR: A PROTECTED_TARGET pointer or its subobject may not be associated with dummy argument 'fptr=' without PROTECTED_TARGET
    call c_f_pointer(cp, p)
  end subroutine

  subroutine implicit_procedures(p, dummy, ptr)
    integer, pointer, protected_target :: p
    procedure() :: dummy
    procedure(), pointer :: ptr
    !ERROR: A PROTECTED_TARGET pointer or its subobject may not be an actual argument to a procedure with an implicit interface
    call dummy(p)
    !ERROR: A PROTECTED_TARGET pointer or its subobject may not be an actual argument to a procedure with an implicit interface
    call ptr(p)
  end subroutine

  subroutine ranks(p)
    integer, pointer, protected_target :: p(..)
    select rank (a => p)
    rank (0)
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
      a = 1
      !ERROR: A PROTECTED_TARGET pointer or its subobject may not be associated with dummy argument 'x=' without PROTECTED_TARGET
      call pointer_in(a)
      call protected_pointer(a)
    rank (1)
      !ERROR: A PROTECTED_TARGET pointer or its subobject may not be associated with dummy argument 'x=' without PROTECTED_TARGET
      call pointer_array(a)
      !ERROR: A PROTECTED_TARGET pointer or its subobject may be associated with nonpointer dummy argument 'x=' only with INTENT(IN)
      call default_array(a)
      call read_rank(a)
    rank default
      !ERROR: A PROTECTED_TARGET pointer or its subobject may be associated with nonpointer dummy argument 'x=' only with INTENT(IN)
      call default_rank(a)
      call read_rank(a)
    end select
  end subroutine

  subroutine types(p)
    class(t), pointer, protected_target :: p
    select type (a => p)
    type is (t)
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
      a%value = 1
      call value_in(a%value)
      call value_default(a%value)
    end select
  end subroutine

  subroutine protected_pointer(x)
    integer, pointer, protected_target, intent(in) :: x
  end subroutine
  subroutine protected_pointer_out(x)
    integer, pointer, protected_target, intent(out) :: x
    nullify(x)
  end subroutine
  subroutine pointer_in(x)
    integer, pointer, intent(in) :: x
  end subroutine
  subroutine pointer_out(x)
    integer, pointer, intent(out) :: x
  end subroutine
  subroutine pointer_array(x)
    integer, pointer, intent(in) :: x(:)
  end subroutine
  subroutine default_array(x)
    integer :: x(:)
  end subroutine
  elemental subroutine elemental_read(x)
    integer, intent(in) :: x
  end subroutine
  subroutine value_in(x)
    integer, intent(in) :: x
  end subroutine
  subroutine value_default(x)
    integer :: x
  end subroutine
  subroutine value_out(x)
    integer, intent(out) :: x
  end subroutine
  subroutine value_inout(x)
    integer, intent(inout) :: x
  end subroutine
  subroutine value_copy(x)
    integer, value :: x
  end subroutine
  subroutine read_value(self)
    class(t), intent(in) :: self
  end subroutine
  subroutine default_value(self)
    class(t) :: self
  end subroutine
  subroutine read_rank(x)
    integer, intent(in) :: x(..)
  end subroutine
  subroutine default_rank(x)
    integer :: x(..)
  end subroutine
  function result_pointer() result(res)
    integer, pointer, protected_target :: res
    integer, target, save :: x = 1
    res => x
  end function
end module

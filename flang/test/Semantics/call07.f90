! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
! Test 15.5.2.7 constraints and restrictions for POINTER dummy arguments.

module m
  real :: coarray(10)[*]
 contains

  subroutine s01(p)
    real, pointer, contiguous, intent(in) :: p(:)
  end subroutine
  subroutine s02(p)
    real, pointer :: p(:)
  end subroutine
  subroutine s03(p)
    real, pointer, intent(in) :: p(:)
  end subroutine
  subroutine s04(p)
    real, pointer :: p
  end subroutine

  subroutine test
    !PORTABILITY: CONTIGUOUS entity 'a01' should be an array pointer, assumed-shape, or assumed-rank [-Wredundant-contiguous]
    real, pointer, contiguous :: a01 ! C830
    real, pointer :: a02(:)
    real, target :: a03(10)
    real :: a04(10) ! not TARGET
    !PORTABILITY: CONTIGUOUS entity 'scalar' should be an array pointer, assumed-shape, or assumed-rank [-Wredundant-contiguous]
    real, contiguous :: scalar
    call s01(a03) ! ok
    !ERROR: CONTIGUOUS pointer dummy argument may not be associated with non-CONTIGUOUS pointer actual argument
    call s01(a02)
    !WARNING: Target of CONTIGUOUS pointer association is not known to be contiguous [-Wpointer-to-possible-noncontiguous]
    call s01(a02(:))
    !ERROR: CONTIGUOUS pointer may not be associated with a discontiguous target
    call s01(a03(::2))
    call s02(a02) ! ok
    call s03(a03) ! ok
    !ERROR: Actual argument associated with POINTER dummy argument 'p=' must also be POINTER unless INTENT(IN)
    call s02(a03)
    !ERROR: Actual argument associated with POINTER dummy argument 'p=' must also be POINTER unless INTENT(IN)
    call s04(a02(1))
    !ERROR: An array section with a vector subscript may not be a pointer target
    call s03(a03([1,2,4]))
    !ERROR: A coindexed object may not be a pointer target
    call s03(coarray(:)[1])
    !ERROR: Target associated with dummy argument 'p=' must be a designator or a call to a pointer-valued function
    call s03([1.])
    !ERROR: In assignment to object dummy argument 'p=', the target 'a04' is not an object with POINTER or TARGET attributes
    call s03(a04)
  end subroutine
end module

! J3/26-007r1 C872-C874.
module protected_target_calls
  type node
    real :: value
    real, pointer :: link
  end type
  real, target :: storage
  integer, protected :: substring_bound = 1
contains
  function readonly_result() result(p)
    real, pointer, protected_target :: p
    p => storage
  end function
  subroutine pointer_in(p)
    real, pointer, protected_target, intent(in) :: p
  end subroutine
  subroutine pointer_out(p)
    real, pointer, protected_target, intent(out) :: p
  end subroutine
  subroutine ordinary_pointer(p)
    real, pointer, intent(in) :: p
  end subroutine
  subroutine read_value(x)
    real, intent(in) :: x
  end subroutine
  subroutine write_value(x)
    real, intent(out) :: x
  end subroutine
  subroutine unspecified(x)
    real :: x
  end subroutine
  subroutine read_array(x)
    real, intent(in) :: x(:)
  end subroutine
  subroutine ordinary_assumed_rank(p)
    real, pointer, intent(out) :: p(..)
  end subroutine
  subroutine protected_assumed_rank(p)
    real, pointer, protected_target, intent(out) :: p(..)
  end subroutine
  subroutine unspecified_assumed_rank(x)
    real :: x(..)
  end subroutine
  subroutine write_assumed_rank(x)
    real, intent(out) :: x(..)
  end subroutine
  subroutine unspecified_character(x)
    character(*) :: x
  end subroutine
  subroutine write_character(x)
    character(*), intent(out) :: x
  end subroutine
  subroutine test_calls(p, a)
    real, pointer, protected_target :: p, a(:)
    real, pointer :: ordinary
    real :: value
    logical :: associated_value
    integer :: count
    external implicit
    call pointer_in(p)
    call pointer_out(p) ! Only association status is being defined.
    call pointer_in(ordinary)
    call pointer_out(ordinary)
    call read_value(p)
    call read_value(a(1))
    call read_array(a(2:))
    call unspecified(p+0.) ! An expression is not a protected subobject.
    call unspecified((p))
    call read_value(readonly_result())
    !ERROR: Target 'p' with PROTECTED_TARGET attribute may not be associated with dummy argument 'p=' without PROTECTED_TARGET
    call ordinary_pointer(readonly_result())
    call pointer_in(readonly_result())
    !ERROR: A PROTECTED_TARGET pointer or its subobject may be associated with nonpointer dummy argument 'x=' only if it has INTENT(IN)
    call unspecified(readonly_result())
    value = sum(a)
    count = size(a)
    associated_value = associated(p)
    !ERROR: Target 'p' with PROTECTED_TARGET attribute may not be associated with dummy argument 'p=' without PROTECTED_TARGET
    call ordinary_pointer(p)
    !ERROR: Target 'a' with PROTECTED_TARGET attribute may not be associated with dummy argument 'p=' without PROTECTED_TARGET
    call ordinary_pointer(a(1))
    !ERROR: A PROTECTED_TARGET pointer or its subobject may be associated with nonpointer dummy argument 'x=' only if it has INTENT(IN)
    call unspecified(p)
    !ERROR: A PROTECTED_TARGET pointer or its subobject may be associated with nonpointer dummy argument 'x=' only if it has INTENT(IN)
    !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'x=' is not definable
    !BECAUSE: 'p' has the PROTECTED_TARGET attribute and its target is not definable
    call write_value(p)
    !ERROR: A PROTECTED_TARGET pointer or its subobject may not be an actual argument to a procedure with an implicit interface
    call implicit(p)
    !ERROR: A PROTECTED_TARGET pointer or its subobject may not be an actual argument to a procedure with an implicit interface
    call implicit(a(1))
    !ERROR: A PROTECTED_TARGET pointer or its subobject may be associated with nonpointer dummy argument 'harvest=' only if it has INTENT(IN)
    !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'harvest=' is not definable
    !BECAUSE: 'p' has the PROTECTED_TARGET attribute and its target is not definable
    call random_number(p)
    associate(q => p)
      call read_value(q)
      call unspecified(q)
      call implicit(q)
      !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'x=' is not definable
      !BECAUSE: 'p' has the PROTECTED_TARGET attribute and its target is not definable
      call write_value(q)
    end associate
  end subroutine
  subroutine subobjects(parent)
    class(node), pointer, protected_target :: parent
    call read_value(parent%value)
    call unspecified(parent%link)
    !ERROR: Target 'parent' with PROTECTED_TARGET attribute may not be associated with dummy argument 'p=' without PROTECTED_TARGET
    call ordinary_pointer(parent%value)
    select type(q => parent)
    type is(node)
      call read_value(q%value)
      call unspecified(q%link)
      call unspecified(q%value)
      !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'x=' is not definable
      !BECAUSE: 'parent' has the PROTECTED_TARGET attribute and its target is not definable
      call write_value(q%value)
    end select
  end subroutine
  subroutine ranked_calls(p)
    real, pointer, protected_target :: p(..)
    external implicit
    select rank(q => p)
    rank(0)
      !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
      !BECAUSE: 'p' has the PROTECTED_TARGET attribute and its target is not definable
      call ordinary_assumed_rank(q)
      call protected_assumed_rank(q)
      call unspecified(q)
      call implicit(q)
      !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'x=' is not definable
      !BECAUSE: 'p' has the PROTECTED_TARGET attribute and its target is not definable
      call write_value(q)
    rank(1)
      !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
      !BECAUSE: 'p' has the PROTECTED_TARGET attribute and its target is not definable
      call ordinary_assumed_rank(q)
      call protected_assumed_rank(q)
      call unspecified_assumed_rank(q)
      !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'x=' is not definable
      !BECAUSE: 'p' has the PROTECTED_TARGET attribute and its target is not definable
      call write_assumed_rank(q)
    rank default
      !ERROR: Target 'q' with PROTECTED_TARGET attribute may not be associated with dummy argument 'p=' without PROTECTED_TARGET
      call ordinary_assumed_rank(q)
      call protected_assumed_rank(q)
      !ERROR: A PROTECTED_TARGET pointer or its subobject may be associated with nonpointer dummy argument 'x=' only if it has INTENT(IN)
      call unspecified_assumed_rank(q)
      !ERROR: A PROTECTED_TARGET pointer or its subobject may be associated with nonpointer dummy argument 'x=' only if it has INTENT(IN)
      !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'x=' is not definable
      !BECAUSE: 'p' has the PROTECTED_TARGET attribute and its target is not definable
      call write_assumed_rank(q)
    end select
  end subroutine
  subroutine substring_calls(index)
    integer, pointer, protected_target :: index
    character(4), target :: text
    external implicit_text
    call unspecified_character(text(index:index))
    call write_character(text(index:index))
    associate(q => text(index:index))
      call unspecified_character(q)
      call implicit_text(q)
      call write_character(q)
    end associate
  end subroutine
end module

subroutine protected_bound_calls
  use protected_target_calls, only: substring_bound, &
    unspecified_character, write_character
  character(4) :: text
  external implicit_text
  associate(q => text(substring_bound:substring_bound))
    call unspecified_character(q)
    call implicit_text(q)
    call write_character(q)
  end associate
end subroutine

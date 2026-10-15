! RUN: %python %S/test_errors.py %s %flang_fc1 -Wpointer-to-undefinable
! F2028 PROTECTED_TARGET: association of the target of a PROTECTED_TARGET
! pointer with other pointers and with dummy arguments (J3/26-007r1 C870-C874)
module m1
  real, protected, target :: x = 1.
end

module m2
  use, intrinsic :: iso_c_binding, only: c_loc
  type t
    real, pointer :: q
  end type
  type tpt
    real, pointer, protected_target :: q
  end type
  type t2
    real :: x
    real, pointer :: q
  end type
  type tb
   contains
    procedure, nopass :: getp => pf
  end type
  real, target :: tgt, tgtarr(2)
  real, pointer, protected_target :: p, parr(:)
  type(t2), pointer, protected_target :: tp
  real, pointer :: w, warr(:)
  interface
    subroutine ptr_in(a)
      real, pointer, intent(in) :: a
    end
    subroutine pt_in(a)
      real, pointer, protected_target, intent(in) :: a
    end
    subroutine pt_out(a)
      real, pointer, protected_target, intent(out) :: a
    end
    subroutine ptr_inout(a)
      real, pointer, intent(in out) :: a
    end
    subroutine arr_ptr_in(a)
      real, pointer, intent(in) :: a(:)
    end
    subroutine obj_in(a)
      real, intent(in) :: a
    end
    subroutine obj_default(a)
      real :: a
    end
    subroutine obj_value(a)
      real, value :: a
    end
    subroutine obj_out(a)
      real, intent(out) :: a
    end
    subroutine obj_inout(a)
      real, intent(in out) :: a
    end
    subroutine arr_in(a)
      real, intent(in) :: a(:)
    end
  end interface
  interface gen
    procedure obj_default
  end interface
 contains
  function pf()
    real, pointer, protected_target :: pf
    pf => tgt
  end
  function wf()
    real, pointer :: wf
    wf => tgt
  end
  subroutine assignments(xin)
    use m1
    real, intent(in), target :: xin
    type(tb) :: obj
    ! F2028 8.5.16 NOTE 1 shows this association as valid, although C864
    ! lists data-target among the prohibited appearances of 'x'; there is
    ! no diagnostic for a PROTECTED_TARGET pointer.
    p => x
    !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
    !BECAUSE: 'x' is protected in this scope
    w => x
    p => xin
    !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
    !BECAUSE: 'xin' is an INTENT(IN) dummy argument
    w => xin
    p => tgt
    w => tgt
    p => w
    p => p
    p => pf()
    p => wf()
    !ERROR: Pointer 'w' must have the PROTECTED_TARGET attribute to be associated with the target of PROTECTED_TARGET pointer 'p'
    w => p
    !ERROR: Pointer 'warr' must have the PROTECTED_TARGET attribute to be associated with the target of PROTECTED_TARGET pointer 'parr'
    warr => parr(1:1)
    !ERROR: Pointer 'w' must have the PROTECTED_TARGET attribute to be associated with the target of PROTECTED_TARGET pointer 'tp'
    w => tp%x
    ! The target of an ordinary pointer component is not part of the
    ! target of 'tp'.
    w => tp%q
    !ERROR: Pointer 'w' must have the PROTECTED_TARGET attribute to be associated with the target of PROTECTED_TARGET pointer 'pf'
    w => pf()
    !ERROR: Pointer 'w' must have the PROTECTED_TARGET attribute to be associated with the target of PROTECTED_TARGET pointer 'pf'
    w => obj%getp()
  end
  subroutine constructors
    type(t) :: a
    type(tpt) :: b
    !ERROR: Pointer 'q' must have the PROTECTED_TARGET attribute to be associated with the target of PROTECTED_TARGET pointer 'p'
    a = t(p)
    !ERROR: Pointer 'q' must have the PROTECTED_TARGET attribute to be associated with the target of PROTECTED_TARGET pointer 'tp'
    a = t(tp%x)
    a = t(w)
    a = t(tp%q)
    b = tpt(p)
    b = tpt(w)
  end
  subroutine calls
    external implicit
    !ERROR: POINTER dummy argument 'a=' must have the PROTECTED_TARGET attribute to be associated with the target of PROTECTED_TARGET pointer 'p'
    call ptr_in(p)
    !ERROR: POINTER dummy argument 'a=' must have the PROTECTED_TARGET attribute to be associated with the target of PROTECTED_TARGET pointer 'p'
    call ptr_inout(p)
    !ERROR: POINTER dummy argument 'a=' must have the PROTECTED_TARGET attribute to be associated with the target of PROTECTED_TARGET pointer 'parr'
    call arr_ptr_in(parr(1:2))
    !ERROR: POINTER dummy argument 'a=' must have the PROTECTED_TARGET attribute to be associated with the target of PROTECTED_TARGET pointer 'pf'
    call ptr_in(pf())
    call pt_in(p)
    call pt_in(pf())
    call pt_out(p)
    call ptr_in(w)
    call pt_in(w)
    call obj_in(p)
    call arr_in(parr(1:2))
    !ERROR: Nonpointer dummy argument 'a=' must have INTENT(IN) to be associated with the target of PROTECTED_TARGET pointer 'p'
    call obj_default(p)
    !ERROR: Nonpointer dummy argument 'a=' must have INTENT(IN) to be associated with the target of PROTECTED_TARGET pointer 'p'
    call obj_value(p)
    !ERROR: Nonpointer dummy argument 'a=' must have INTENT(IN) to be associated with the target of PROTECTED_TARGET pointer 'tp'
    call obj_default(tp%x)
    !ERROR: Nonpointer dummy argument 'a=' must have INTENT(IN) to be associated with the target of PROTECTED_TARGET pointer 'pf'
    call obj_default(pf())
    !ERROR: Nonpointer dummy argument 'a=' must have INTENT(IN) to be associated with the target of PROTECTED_TARGET pointer 'p'
    call gen(p)
    !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'a=' is not definable
    !BECAUSE: 'p' is a PROTECTED_TARGET pointer whose target may not be defined through it
    call obj_out(p)
    !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'a=' is not definable
    !BECAUSE: 'p' is a PROTECTED_TARGET pointer whose target may not be defined through it
    call obj_inout(p)
    ! Neither the target of an ordinary pointer component nor the value of
    ! an expression is part of the target of a PROTECTED_TARGET pointer.
    call obj_default(tp%q)
    call obj_default((p))
    call obj_default(p + 1.)
    !ERROR: The target of PROTECTED_TARGET pointer 'p' may not be an actual argument to a procedure with an implicit interface
    call implicit(p)
    !ERROR: The target of PROTECTED_TARGET pointer 'tp' may not be an actual argument to a procedure with an implicit interface
    call implicit(tp%x)
    !ERROR: The target of PROTECTED_TARGET pointer 'pf' may not be an actual argument to a procedure with an implicit interface
    call implicit(pf())
    call implicit(tp%q)
    call implicit((p))
    ! Intrinsic procedures are exempt from the C872 and C874 checks; their
    ! INTENT(OUT) arguments are still checked for definability.
    print *, associated(p), associated(p, tgt), size(parr), c_loc(p)
    !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'harvest=' is not definable
    !BECAUSE: 'p' is a PROTECTED_TARGET pointer whose target may not be defined through it
    call random_number(p)
  end
end

module m5
  use m1
  ! The same decision applies to an initial-data-target.
  real, pointer, protected_target :: pinit => x
  !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
  !BECAUSE: 'x' is protected in this scope
  real, pointer :: winit => x
end

module m6
  use m2, only: z => p, tgt
 contains
  subroutine renamed
    real, pointer, protected_target :: q
    real, pointer :: w
    z => tgt
    q => z
    !ERROR: Pointer 'w' must have the PROTECTED_TARGET attribute to be associated with the target of PROTECTED_TARGET pointer 'p'
    w => z
  end
end

module m7
  use m2, only: p, parr, pf, w
  type t
    real :: x
  end type
  interface assignment(=)
    module procedure asgn_value, asgn_to_real
  end interface
  interface operator(+)
    module procedure add_value
  end interface
 contains
  subroutine asgn_value(a, b)
    type(t), intent(out) :: a
    real, value :: b
  end
  subroutine asgn_to_real(a, b)
    real, intent(out) :: a
    type(t), intent(in) :: b
  end
  function add_value(a, b)
    type(t), intent(in) :: a
    real, value :: b
    type(t) :: add_value
  end
  subroutine defined
    type(t) :: tv
    ! The right-hand side of a defined assignment is passed as if it were
    ! enclosed in parentheses.
    tv = p
    tv = parr(1)
    tv = pf()
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'p' is a PROTECTED_TARGET pointer whose target may not be defined through it
    !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'a=' is not definable
    !BECAUSE: 'p' is a PROTECTED_TARGET pointer whose target may not be defined through it
    p = tv
    !ERROR: Nonpointer dummy argument 'b=' must have INTENT(IN) to be associated with the target of PROTECTED_TARGET pointer 'p'
    call asgn_value(tv, p)
    ! The operands of a defined operation are not.
    !ERROR: Nonpointer dummy argument 'b=' must have INTENT(IN) to be associated with the target of PROTECTED_TARGET pointer 'p'
    tv = tv + p
    tv = tv + (p)
  end
  subroutine statement_function
    real :: sf, a, y
    sf(a) = a + 1.
    !ERROR: The target of PROTECTED_TARGET pointer 'p' may not be an actual argument to a procedure with an implicit interface
    y = sf(p)
    y = sf((p))
    y = sf(w)
  end
end

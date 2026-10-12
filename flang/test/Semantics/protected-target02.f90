! RUN: %python %S/test_errors.py %s %flang_fc1 -Wpointer-to-undefinable
! F2028 C866, C870, C871 and 8.5.16 p2.

module protected_target_objects
  type t
    integer :: n
    integer, pointer :: link
    procedure(), pointer, nopass :: proc
  end type
  type read_pointer
    integer, pointer, protected_target :: p
  end type
  type write_pointer
    integer, pointer :: p
  end type
  integer, target :: x = 1
  integer, target :: array(2)
  integer, protected, target :: protected_x
  type(t), protected, target :: protected_t
contains
  subroutine definitions(p, a, c, z, obj)
    integer, pointer, protected_target :: p, a(:)
    character(:), pointer, protected_target :: c
    complex, pointer, protected_target :: z
    type(t), pointer, protected_target :: obj
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable
    p = 2
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'a' is not definable
    if (.false.) a(1:2) = 3
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'c' is not definable
    c(1:1) = 'a'
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'z' is not definable
    z%re = 1.
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'obj' is not definable
    obj%n = 2
    !ERROR: Input variable 'p' is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable
    read *, p
    !ERROR: Internal file variable 'c' is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'c' is not definable
    write(c, *) 1
    !ERROR: 'p' may not be used as a DO variable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable
    do p = 1, 2
    end do
    !ERROR: The left-hand side of a pointer assignment is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'obj' is not definable
    obj%link => x
    !ERROR: 'link' may not appear in NULLIFY
    !BECAUSE: Target of PROTECTED_TARGET pointer 'obj' is not definable
    nullify(obj%link)
    !ERROR: The left-hand side of a pointer assignment is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'obj' is not definable
    obj%proc => noop
    ! The target of a different pointer is not a subobject (F2028 9.4.2 p5).
    obj%link = 2
    p => x
    x = 2
    nullify(p)
    ! Defining an associate name defines its selector (F2028 20.6.7(11)).
    associate (alias => a(1:2))
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: Target of PROTECTED_TARGET pointer 'a' is not definable
      alias = 4
    end associate
  end subroutine

  subroutine polymorphic(p)
    class(t), pointer, protected_target :: p
    integer, pointer :: w
    select type (q => p)
    type is (t)
      !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
      !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable
      w => q%n
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable
      q%n = 1
    end select
  end subroutine

  subroutine assumed_rank(p)
    integer, pointer, protected_target :: p(..)
    integer, pointer :: w(:)
    select rank(q => p)
    rank(1)
      !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
      !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable
      w => q
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable
      q(1) = 1
    end select
  end subroutine

  subroutine propagation(p, a, obj)
    integer, pointer, protected_target :: p, a(:), r, ra(:)
    type(t), pointer, protected_target :: obj
    integer, pointer :: w, wa(:)
    type(read_pointer) :: rp
    type(write_pointer) :: wp
    !ERROR: PROTECTED_TARGET pointer 'p' or a subobject may not be associated with pointer 'w' without PROTECTED_TARGET
    w => p
    !ERROR: PROTECTED_TARGET pointer 'a' or a subobject may not be associated with pointer 'wa' without PROTECTED_TARGET
    wa => a(1:2)
    !ERROR: PROTECTED_TARGET pointer 'obj' or a subobject may not be associated with pointer 'w' without PROTECTED_TARGET
    w => obj%n
    !ERROR: PROTECTED_TARGET pointer 'p' or a subobject may not be associated with pointer 'p' without PROTECTED_TARGET
    wp = write_pointer(p)
    !ERROR: PROTECTED_TARGET pointer 'obj' or a subobject may not be associated with pointer 'p' without PROTECTED_TARGET
    wp = write_pointer(obj%n)
    rp = read_pointer(p)
    rp = read_pointer(obj%n)
    w => x
    r => w
    r => p
    ra(1:2) => a(1:2)
    r => null()
    ! A data-target reads obj%link's association, not its protected container.
    w => obj%link
    wp = write_pointer(obj%link)
    ! A protected pointer used only as a subscript does not protect array.
    wa => array(p:p)
    ! Associate names do not have PROTECTED_TARGET (F2028 C870/C871).
    ! These associations are accepted with only the optional warning.
    associate (alias => p)
      !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
      !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable
      w => alias
      !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
      !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable
      wp = write_pointer(alias)
    end associate
  end subroutine

  function get() result(p)
    integer, pointer, protected_target :: p
    p => x
  end function

  subroutine function_targets
    integer, pointer, protected_target :: p
    integer, pointer :: w
    procedure(get), pointer :: f
    p => get()
    !ERROR: The PROTECTED_TARGET result of 'get' or a subobject may not be associated with pointer 'w' without PROTECTED_TARGET
    w => get()
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: Target of the PROTECTED_TARGET result of 'get' is not definable
    get() = 1
    associate (q => get())
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: Target of the PROTECTED_TARGET result of 'get' is not definable
      q = 1
    end associate
    f => get
    !ERROR: The PROTECTED_TARGET result of 'f' or a subobject may not be associated with pointer 'w' without PROTECTED_TARGET
    w => f()
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: Target of the PROTECTED_TARGET result of 'f' is not definable
    f() = 1
  end subroutine

  subroutine intent_in(p, ordinary, target)
    integer, pointer, protected_target, intent(in) :: p
    integer, pointer, intent(in) :: ordinary
    integer, target, intent(in) :: target
    integer, pointer, protected_target :: r
    ordinary = 1
    r => target
    !ERROR: The left-hand side of a pointer assignment is not definable
    !BECAUSE: 'p' is an INTENT(IN) dummy argument
    p => x
    !ERROR: 'p' may not appear in NULLIFY
    !BECAUSE: 'p' is an INTENT(IN) dummy argument
    nullify(p)
  end subroutine

  subroutine owner
    integer, pointer, protected_target :: r
    r => protected_x
    call read_protected_pointer(protected_x)
  end subroutine
  subroutine read_protected_pointer(q)
    integer, pointer, protected_target, intent(in) :: q
  end subroutine
  subroutine noop
  end subroutine
end module

subroutine outside
  use protected_target_objects
  integer, pointer, protected_target :: r
  integer, pointer :: ordinary
  type(read_pointer) :: rp
  ! F2028 C864 is normative; its conflicting 8.5.16 NOTE 1 is not followed.
  !ERROR: Use-associated PROTECTED object 'protected_x' may not be a target of a PROTECTED_TARGET pointer
  r => protected_x
  !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
  !BECAUSE: 'protected_x' is protected in this scope
  ordinary => protected_x
  !ERROR: Use-associated PROTECTED object 'protected_t' may not be a target of a PROTECTED_TARGET pointer
  r => protected_t%n
  !ERROR: Use-associated PROTECTED object 'protected_x' may not be a target of a PROTECTED_TARGET pointer
  rp = read_pointer(protected_x)
  ! F2028 15.5.2.8 p2 requires a valid pointer-assignment target for this actual.
  !ERROR: Use-associated PROTECTED object 'protected_x' may not be a target of a PROTECTED_TARGET pointer
  call read_protected_pointer(protected_x)
  r => protected_t%link
end subroutine

module protected_initial_target
  use protected_target_objects, only: protected_x, protected_t
  !ERROR: Use-associated PROTECTED object 'protected_x' may not be a target of a PROTECTED_TARGET pointer
  integer, pointer, protected_target :: initial => protected_x
  !ERROR: Use-associated PROTECTED object 'protected_t' may not be a target of a PROTECTED_TARGET pointer
  integer, pointer, protected_target :: initial_subobject => protected_t%n
  type initialized_component
    !ERROR: Use-associated PROTECTED object 'protected_x' may not be a target of a PROTECTED_TARGET pointer
    integer, pointer, protected_target :: p => protected_x
  end type
end module

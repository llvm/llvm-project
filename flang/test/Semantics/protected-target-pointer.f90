! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
! The target of a PROTECTED_TARGET pointer may become associated only with
! another PROTECTED_TARGET pointer (F'2028 C870, C871).

module m
  integer, target, protected :: px = 1
  integer, pointer, protected_target :: mp
  type :: t
    integer, pointer :: c
  end type
  type :: pt_t
    integer, pointer, protected_target :: c
  end type
  type :: node
    integer :: n
    integer, pointer :: next
   contains
    procedure :: getn
  end type
 contains
  function get() result(r)
    integer, pointer, protected_target :: r
    r => px
  end
  function getn(this) result(r)
    class(node), intent(in) :: this
    integer, pointer, protected_target :: r
    r => this%next
  end
  function getw() result(r)
    integer, pointer :: r
    r => px
  end
end

subroutine assignments(p, pa, pn, x)
  use m
  integer, pointer, protected_target :: p, pa(:), q, qa(:)
  type(node), pointer, protected_target :: pn
  integer, target :: x
  integer, pointer :: w, wa(:)
  type(t) :: a
  type(pt_t) :: b
  procedure(get), pointer :: fp
  !ERROR: The target of PROTECTED_TARGET pointer 'p' may not be associated with pointer 'w', which does not have the PROTECTED_TARGET attribute
  w => p
  !ERROR: The target of PROTECTED_TARGET pointer 'pa' may not be associated with pointer 'w', which does not have the PROTECTED_TARGET attribute
  w => pa(1)
  !ERROR: The target of PROTECTED_TARGET pointer 'pa' may not be associated with pointer 'wa', which does not have the PROTECTED_TARGET attribute
  wa => pa(1:2)
  !ERROR: The target of PROTECTED_TARGET pointer 'pa' may not be associated with pointer 'wa', which does not have the PROTECTED_TARGET attribute
  wa(1:2) => pa
  !ERROR: The target of PROTECTED_TARGET pointer 'pn' may not be associated with pointer 'w', which does not have the PROTECTED_TARGET attribute
  w => pn%n
  !ERROR: The target of PROTECTED_TARGET pointer 'r' may not be associated with pointer 'w', which does not have the PROTECTED_TARGET attribute
  w => get()
  !ERROR: The target of PROTECTED_TARGET pointer 'r' may not be associated with pointer 'w', which does not have the PROTECTED_TARGET attribute
  w => pn%getn()
  fp => get
  !ERROR: The target of PROTECTED_TARGET pointer 'r' may not be associated with pointer 'w', which does not have the PROTECTED_TARGET attribute
  w => fp()
  !ERROR: The target of PROTECTED_TARGET pointer 'c' may not be associated with pointer 'w', which does not have the PROTECTED_TARGET attribute
  w => b%c
  !ERROR: The target of PROTECTED_TARGET pointer 'p' may not be associated with pointer 'c', which does not have the PROTECTED_TARGET attribute
  a%c => p
  ! Accepted: an ASSOCIATE name does not have the PROTECTED_TARGET attribute
  ! (F'2028 11.1.3.3p1), so only the optional warning applies.
  associate (s => pa)
    !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
    !BECAUSE: 'pa' has the PROTECTED_TARGET attribute
    w => s(1)
  end associate
  !ERROR: The target of PROTECTED_TARGET pointer 'p' may not be associated with pointer 'c', which does not have the PROTECTED_TARGET attribute
  a = t(p)
  !ERROR: The target of PROTECTED_TARGET pointer 'pa' may not be associated with pointer 'c', which does not have the PROTECTED_TARGET attribute
  a = t(c=pa(2))
  ! A pointer component of the target is a subobject of the target as a
  ! data-target, but its own target is not.
  !ERROR: Pointer component 'next' of the target of PROTECTED_TARGET pointer 'pn' may not be a data-target for pointer 'w', which does not have the PROTECTED_TARGET attribute
  w => pn%next
  !ERROR: Pointer component 'next' of the target of PROTECTED_TARGET pointer 'pn' may not be a data-target for pointer 'c', which does not have the PROTECTED_TARGET attribute
  a%c => pn%next
  !ERROR: Pointer component 'next' of the target of PROTECTED_TARGET pointer 'pn' may not be a data-target for pointer 'c', which does not have the PROTECTED_TARGET attribute
  a = t(pn%next)
  associate (s => pn%next)
    w => s
  end associate
  ! Valid: protection is kept, or an ordinary pointer gains it.
  q => p
  q => pa(1)
  qa => pa(1:2)
  q => pn%n
  q => pn%next
  q => get()
  q => pn%getn()
  q => fp()
  q => getw()
  q => b%c
  q => w
  b%c => p
  b%c => pn%next
  b = pt_t(p)
  b = pt_t(c=pa(1))
  b = pt_t(pn%next)
  a = t(w)
  w => x
  q => x
  print *, q, qa, a%c, b%c
end

subroutine select_rank(ar, qa)
  use m
  integer, pointer, protected_target :: ar(..), qa(:)
  integer, pointer :: w(:)
  type(t) :: a
  ! A RANK(n) associate name is a pointer without the PROTECTED_TARGET
  ! attribute (F'2028 11.1.12.3p3).  Accepted as a data-target: only the
  ! optional warning applies.
  select rank (r => ar)
  rank (0)
    !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
    !BECAUSE: 'ar' has the PROTECTED_TARGET attribute
    a = t(r)
  rank (1)
    !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
    !BECAUSE: 'ar' has the PROTECTED_TARGET attribute
    w => r
    !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
    !BECAUSE: 'ar' has the PROTECTED_TARGET attribute
    w => r(1:1)
    !ERROR: The target of PROTECTED_TARGET pointer 'qa' may not be associated with pointer 'r', which does not have the PROTECTED_TARGET attribute
    r => qa
    qa => r
  rank default
    ! A RANK DEFAULT associate name has exactly the attributes of its
    ! selector (F'2028 11.1.12.3p2).
    !ERROR: The target of PROTECTED_TARGET pointer 'ar' may not be associated with pointer 'w', which does not have the PROTECTED_TARGET attribute
    w(1:1) => r
    qa(1:1) => r
  end select
end

subroutine targets(x)
  use m
  integer, target, intent(in) :: x
  integer, pointer :: w
  integer, pointer, protected_target :: q
  ! F'2028 C864 forbids a use-associated nonpointer PROTECTED object as a
  ! data-target, although the non-normative 8.5.16 NOTE 1 shows the
  ! equivalent of "mp => px" as valid.  For every pointer, Flang reports C864
  ! only as an optional warning.  An INTENT(IN) target violates no
  ! constraint, and a PROTECTED_TARGET pointer cannot define it, so it gets
  ! no warning.
  !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
  !BECAUSE: 'px' is protected in this scope
  q => px
  !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
  !BECAUSE: 'px' is protected in this scope
  mp => px
  q => x
  !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
  !BECAUSE: 'px' is protected in this scope
  w => px
  !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
  !BECAUSE: 'x' is an INTENT(IN) dummy argument
  w => x
  print *, q, w
end

module m2
  use m
  ! Pointer initialization is treated like the pointer assignments above:
  ! C864 also forbids px as an initial-data-target.
  !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
  !BECAUSE: 'px' is protected in this scope
  integer, pointer, protected_target :: init => px
  !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
  !BECAUSE: 'px' is protected in this scope
  integer, pointer :: winit => px
  type(node), pointer, protected_target :: mpn
  ! An initial data target may not involve a pointer, so C870 is not reached.
  !ERROR: An initial data target may not be a reference to a POINTER 'next'
  integer, pointer :: wnext => mpn%next
end

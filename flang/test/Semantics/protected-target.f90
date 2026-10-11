! RUN: %python %S/test_errors.py %s %flang_fc1

subroutine c8109_stmt()
  !ERROR: A PROTECTED_TARGET entity must be a data pointer
  integer :: i
  protected_target :: i
end

subroutine attr_spec_nonpointer()
  !ERROR: A PROTECTED_TARGET entity must be a data pointer
  integer, protected_target :: i
  !ERROR: A PROTECTED_TARGET entity must be a data pointer
  integer, allocatable, protected_target :: a(:)
end

subroutine c781_component()
  type t
    !ERROR: A PROTECTED_TARGET entity must be a data pointer
    integer, protected_target :: c
  end type
end

subroutine c867_common()
  !ERROR: A pointer with the PROTECTED_TARGET attribute may not be in a common block
  integer, pointer, protected_target :: p
  common /blk/ p
end

subroutine c869_allocate()
  integer, pointer, protected_target :: p
  !ERROR: ALLOCATE of PROTECTED_TARGET pointer 'p' requires SOURCE=
  allocate(p)
  !ERROR: ALLOCATE of PROTECTED_TARGET pointer 'p' requires SOURCE=
  allocate(p, mold=1)
end

subroutine c868_deallocate()
  integer, pointer, protected_target :: p
  allocate(p, source=1)
  !ERROR: PROTECTED_TARGET pointer 'p' may not be deallocated
  deallocate(p)
end

subroutine c866_define()
  integer, target :: x = 1
  integer, pointer, protected_target :: p
  p => x
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'p' is a PROTECTED_TARGET pointer, so its target may not be defined through it
  p = 2
end

subroutine c870_drop()
  integer, target :: x = 1
  integer, pointer, protected_target :: p
  integer, pointer :: q
  p => x
  !ERROR: Pointer assignment target is a PROTECTED_TARGET pointer or a subobject of one; the left-hand side pointer must also have the PROTECTED_TARGET attribute
  q => p
end

subroutine c871_constructor()
  type t
    integer, pointer :: q
  end type
  type(t) :: a
  integer, target :: x = 1
  integer, pointer, protected_target :: p
  p => x
  !ERROR: Structure constructor component 'q' must have the PROTECTED_TARGET attribute when its pointer target is a PROTECTED_TARGET pointer or a subobject of one
  a = t(p)
end

subroutine c872_and_c874_calls()
  interface
    subroutine takes_plain_ptr(a)
      integer, pointer :: a
    end
    subroutine takes_pt_ptr(a)
      integer, pointer, protected_target :: a
    end
    subroutine takes_no_intent(a)
      integer :: a
    end
    subroutine takes_value(a)
      integer, value :: a
    end
    subroutine takes_value_in(a)
      integer, value, intent(in) :: a
    end
    subroutine takes_in(a)
      integer, intent(in) :: a
    end
  end interface
  integer, target :: x = 1
  integer, pointer, protected_target :: p
  integer, pointer :: q
  p => x
  q => x
  call takes_pt_ptr(q)
  !ERROR: POINTER dummy argument 'a=' must have the PROTECTED_TARGET attribute when the actual argument is a PROTECTED_TARGET pointer or a subobject of one
  call takes_plain_ptr(p)
  !ERROR: Non-POINTER dummy argument 'a=' must have INTENT(IN) when the actual argument is a PROTECTED_TARGET pointer or a subobject of one
  call takes_no_intent(p)
  !ERROR: Non-POINTER dummy argument 'a=' must have INTENT(IN) when the actual argument is a PROTECTED_TARGET pointer or a subobject of one
  call takes_value(p)
  call takes_value_in(p)
  call takes_in(p)
end

subroutine c874_defined_assignment_duplicate()
  type t
    integer :: x = 0
  end type
  interface assignment(=)
    procedure :: set_t
  end interface
  type(t), target :: obj
  type(t), pointer, protected_target :: p
  p => obj
  ! Keep both errors: C874 for the call and 8.5.16p2 for assignment.
  !ERROR: Non-POINTER dummy argument 'lhs=' must have INTENT(IN) when the actual argument is a PROTECTED_TARGET pointer or a subobject of one
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'p' is a PROTECTED_TARGET pointer, so its target may not be defined through it
  p = 1
contains
  subroutine set_t(lhs, rhs)
    type(t), intent(inout) :: lhs
    integer, intent(in) :: rhs
    lhs%x = rhs
  end
end

subroutine c872_subobject_section()
  interface
    subroutine pin(a)
      integer, pointer, intent(in) :: a
    end
    subroutine pinv(a)
      integer, pointer, intent(in) :: a(:)
    end
    subroutine takes_in(a)
      integer, intent(in) :: a
    end
  end interface
  type t
    integer, pointer :: q(:)
  end type
  type(t), target :: n
  type(t) :: o
  integer, target :: x(4) = [1, 2, 3, 4]
  integer, pointer, protected_target :: pv(:)
  type(t), pointer, protected_target :: p
  integer, pointer :: q(:)
  pv => x
  n%q => x
  p => n
  !ERROR: Pointer assignment target is a PROTECTED_TARGET pointer or a subobject of one; the left-hand side pointer must also have the PROTECTED_TARGET attribute
  q => pv(1:2)
  !ERROR: Structure constructor component 'q' must have the PROTECTED_TARGET attribute when its pointer target is a PROTECTED_TARGET pointer or a subobject of one
  o = t(pv(1:2))
  call takes_in(pv(2))
  !ERROR: POINTER dummy argument 'a=' must have the PROTECTED_TARGET attribute when the actual argument is a PROTECTED_TARGET pointer or a subobject of one
  call pinv(pv(1:2))
  call pin(p%q(1))
  call pinv(p%q(1:2))
  !ERROR: POINTER dummy argument 'a=' must have the PROTECTED_TARGET attribute when the actual argument is a PROTECTED_TARGET pointer or a subobject of one
  call pinv(p%q)
end

subroutine c873_implicit()
  integer, target :: x = 1
  integer, pointer, protected_target :: p
  external ext
  p => x
  !ERROR: Actual argument that is a PROTECTED_TARGET pointer or a subobject of one may not be passed to a procedure with an implicit interface
  call ext(p)
end

subroutine c873_subobject_implicit()
  integer, target :: x(2) = [1, 2]
  integer, pointer, protected_target :: p(:)
  external ext
  p => x
  !ERROR: Actual argument that is a PROTECTED_TARGET pointer or a subobject of one may not be passed to a procedure with an implicit interface
  call ext(p(1))
end

subroutine c874_null_mold()
  integer, pointer, protected_target :: p(:), q(:)
  p => null(mold=q)
  q => null(mold=p)
end

subroutine c868_c869_components()
  type t
    integer, pointer, protected_target :: c
  end type
  type(t) :: x
  !ERROR: ALLOCATE of PROTECTED_TARGET pointer 'c' requires SOURCE=
  allocate(x%c)
  allocate(x%c, source=1)
  !ERROR: PROTECTED_TARGET pointer 'c' may not be deallocated
  deallocate(x%c)
end

subroutine c866_subcomponents()
  type t
    integer, pointer :: q
    integer, allocatable :: a
  end type
  type holder
    integer, pointer :: q
  end type
  interface
    subroutine takes_inout(a)
      integer, intent(inout) :: a
    end
    subroutine noint(a)
      integer, pointer :: a
    end
  end interface
  type(t), pointer, protected_target :: p
  type(holder) :: h
  integer, target :: y
  integer, pointer :: r
  external ext
  p%q = 1
  !ERROR: The left-hand side of a pointer assignment is not definable
  !BECAUSE: 'q' may not be defined via PROTECTED_TARGET pointer 'p'
  p%q => y
  r => p%q
  h = holder(p%q)
  !ERROR: 'q' may not appear in NULLIFY
  !BECAUSE: 'q' may not be defined via PROTECTED_TARGET pointer 'p'
  nullify(p%q)
  !ERROR: Name in ALLOCATE statement is not definable
  !BECAUSE: 'q' may not be defined via PROTECTED_TARGET pointer 'p'
  allocate(p%q)
  !ERROR: Name in ALLOCATE statement is not definable
  !BECAUSE: 'a' may not be defined via PROTECTED_TARGET pointer 'p'
  allocate(p%a)
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'a' may not be defined via PROTECTED_TARGET pointer 'p'
  p%a = 1
  call takes_inout(p%q)
  !ERROR: POINTER dummy argument 'a=' must have the PROTECTED_TARGET attribute when the actual argument is a PROTECTED_TARGET pointer or a subobject of one
  call noint(p%q)
  call ext(p%q)
end

subroutine c870_to_c874_nearest_pointer()
  type leaf
    integer :: c
  end type
  type t
    type(leaf), pointer :: q
    integer :: c
  end type
  type holder
    integer, pointer :: p
  end type
  interface
    subroutine inout(x)
      integer, intent(inout) :: x
    end
  end interface
  type(t), pointer, protected_target :: p
  type(holder) :: h
  integer, pointer :: w
  external ext
  w => p%q%c
  call inout(p%q%c)
  h = holder(p%q%c)
  call ext(p%q%c)
  !ERROR: Pointer assignment target is a PROTECTED_TARGET pointer or a subobject of one; the left-hand side pointer must also have the PROTECTED_TARGET attribute
  w => p%c
  !ERROR: Non-POINTER dummy argument 'x=' must have INTENT(IN) when the actual argument is a PROTECTED_TARGET pointer or a subobject of one
  call inout(p%c)
  !ERROR: Structure constructor component 'p' must have the PROTECTED_TARGET attribute when its pointer target is a PROTECTED_TARGET pointer or a subobject of one
  h = holder(p%c)
  !ERROR: Actual argument that is a PROTECTED_TARGET pointer or a subobject of one may not be passed to a procedure with an implicit interface
  call ext(p%c)
end

subroutine c866_associate()
  interface
    subroutine takes_no_intent_vec(a)
      integer :: a(:)
    end
  end interface
  integer, target :: x(2) = [1, 2]
  integer, pointer, protected_target :: p(:)
  integer, pointer :: q(:)
  p => x
  associate (a => p)
    ! 'a' does not have PROTECTED_TARGET (F'2028 draft 11.1.3.3), so 'q => a' is accepted.
    q => a
    call takes_no_intent_vec(a(1:1))
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'p' is a PROTECTED_TARGET pointer, so its target may not be defined through it
    a(1) = 1
  end associate
end

subroutine c866_associate_pointer_component_selectors()
  type node
    integer, pointer :: q => null()
    integer :: c = 0
  end type
  type(node), target :: n, n2
  type(node), pointer, protected_target :: np
  integer, target :: y = 1
  n%q => y
  np => n
  np%q = 1
  associate (a => np)
    a%q = 2
    !ERROR: The left-hand side of a pointer assignment is not definable
    !BECAUSE: 'np' is a PROTECTED_TARGET pointer, so its target may not be defined through it
    a%q => y
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'np' is a PROTECTED_TARGET pointer, so its target may not be defined through it
    a = n2
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'np' is a PROTECTED_TARGET pointer, so its target may not be defined through it
    a%c = 3
  end associate
end

subroutine c866_associate_component_selectors()
  type t
    integer, pointer, protected_target :: c => null()
    integer, pointer, protected_target :: v(:) => null()
    class(*), pointer, protected_target :: u => null()
  end type
  type plain
    integer, pointer :: c => null()
  end type
  type inner
    integer :: c = 0
  end type
  type holder
    type(inner), pointer, protected_target :: p => null()
  end type
  interface
    subroutine takes_inout(x)
      integer, intent(inout) :: x
    end
  end interface
  type(t) :: o
  type(plain) :: ok
  type(holder) :: h
  integer, target :: y = 1, w(2) = [1, 2]
  type(inner), target :: n
  integer, pointer :: q
  o%c => y
  o%v => w
  o%u => y
  ok%c => y
  h%p => n
  associate (a => ok%c)
    q => a
  end associate
  associate (a => o%c)
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'c' is a PROTECTED_TARGET pointer, so its target may not be defined through it
    a = 2
    !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'x=' is not definable
    !BECAUSE: 'c' is a PROTECTED_TARGET pointer, so its target may not be defined through it
    call takes_inout(a)
  end associate
  associate (b => o%v(1:2))
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'v' is a PROTECTED_TARGET pointer, so its target may not be defined through it
    b(1) = 3
  end associate
  select type (z => o%u)
  type is (integer)
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'u' is a PROTECTED_TARGET pointer, so its target may not be defined through it
    z = 4
  end select
  associate (c => h%p%c)
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'c' may not be defined via PROTECTED_TARGET pointer 'p'
    c = 5
  end associate
end

subroutine c866_select_type_pointer_component_selector()
  type node
    integer, pointer :: q => null()
  end type
  interface
    subroutine takes_inout(v)
      integer, intent(inout) :: v
    end
  end interface
  type(node), target :: n
  class(node), pointer, protected_target :: cp
  integer, target :: y = 1
  n%q => y
  cp => n
  select type (c => cp)
  type is (node)
    c%q = 4
    call takes_inout(c%q)
    !ERROR: The left-hand side of a pointer assignment is not definable
    !BECAUSE: 'cp' is a PROTECTED_TARGET pointer, so its target may not be defined through it
    c%q => y
  end select
end

subroutine c866_associate_pointer_function_selector()
  type node
    integer, pointer :: w => null()
    integer, allocatable :: a
  end type
  interface
    subroutine takes_inout(v)
      integer, intent(inout) :: v
    end
  end interface
  type(node), target :: n
  integer, target :: y = 1
  associate (a => plain_result())
    a%w => y
    nullify(a%w)
    allocate(a%w)
    deallocate(a%w)
    a%a = 1
  end associate
  associate (b => protected_result())
    b%w = 2
    call takes_inout(b%w)
    !ERROR: The left-hand side of a pointer assignment is not definable
    !BECAUSE: 'protected_result()' may not be defined via PROTECTED_TARGET pointer function result
    b%w => y
  end associate
contains
  function plain_result() result(r)
    type(node), pointer :: r
    r => n
  end
  function protected_result() result(r)
    type(node), pointer, protected_target :: r
    r => n
  end
end

subroutine c872_c874_intrinsics()
  use iso_c_binding
  integer, target :: x(2) = [1, 2]
  integer, pointer, protected_target :: p(:)
  integer, pointer, protected_target :: ps(:)
  real, pointer, protected_target :: r(:)
  type(c_ptr) :: cptr
  logical :: l
  integer :: n
  p => x
  l = associated(p)
  l = associated(p, p)
  n = size(p) + lbound(p, 1) + ubound(p, 1)
  ps => null(mold=p)
  cptr = c_loc(x(1))
  !ERROR: POINTER dummy argument 'fptr=' must have the PROTECTED_TARGET attribute when the actual argument is a PROTECTED_TARGET pointer or a subobject of one
  call c_f_pointer(cptr, ps, [2])
  !ERROR: Non-POINTER dummy argument 'harvest=' must have INTENT(IN) when the actual argument is a PROTECTED_TARGET pointer or a subobject of one
  call random_number(r)
end

subroutine c866_definition_contexts()
  implicit none
  integer, target :: x = 1
  integer, pointer, protected_target :: p
  character(4), target :: cx
  character(4), pointer, protected_target :: cp
  integer, allocatable :: a
  integer :: u
  namelist /grp/ p
  p => x
  cp => cx
  !ERROR: Input variable 'p' is not definable
  !BECAUSE: 'p' is a PROTECTED_TARGET pointer, so its target may not be defined through it
  read *, p
  !ERROR: 'p' may not be used as a DO variable
  !BECAUSE: 'p' is a PROTECTED_TARGET pointer, so its target may not be defined through it
  do p = 1, 1
  end do
  !ERROR: IOSTAT variable 'p' is not definable
  !BECAUSE: 'p' is a PROTECTED_TARGET pointer, so its target may not be defined through it
  read(*, *, iostat=p) u
  !ERROR: STAT variable 'p' is not definable
  !BECAUSE: 'p' is a PROTECTED_TARGET pointer, so its target may not be defined through it
  allocate(a, stat=p)
  !ERROR: Internal file variable 'cp' is not definable
  !BECAUSE: 'cp' is a PROTECTED_TARGET pointer, so its target may not be defined through it
  write(cp, '(a)') 'ab'
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'cp' is a PROTECTED_TARGET pointer, so its target may not be defined through it
  cp(1:2) = 'cd'
  !ERROR: NAMELIST input group must not contain undefinable item 'p'
  !BECAUSE: 'p' is a PROTECTED_TARGET pointer, so its target may not be defined through it
  read(*, nml=grp)
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'p' is a PROTECTED_TARGET pointer, so its target may not be defined through it
  if (.false.) p = 2
end

subroutine c866_intent_in_pointer_dummy(p)
  integer, target :: x = 1
  integer, pointer, protected_target, intent(in) :: p
  print *, p
  !ERROR: The left-hand side of a pointer assignment is not definable
  !BECAUSE: 'p' is an INTENT(IN) dummy argument
  p => x
end

subroutine c866_intent_out_pointer_dummy(a)
  integer, pointer, protected_target, intent(out) :: a
  nullify(a)
end

subroutine c872_c874_select_rank_default(ar)
  integer, pointer, protected_target :: ar(..)
  integer, pointer :: q(:)
  interface
    subroutine plainptrar_rank(a)
      integer, pointer :: a(..)
    end
    subroutine noint_rank(a)
      integer :: a(..)
    end
    subroutine ptar_rank(a)
      integer, pointer, protected_target :: a(..)
    end
  end interface
  !ERROR: POINTER dummy argument 'a=' must have the PROTECTED_TARGET attribute when the actual argument is a PROTECTED_TARGET pointer or a subobject of one
  call plainptrar_rank(ar)
  !ERROR: Non-POINTER dummy argument 'a=' must have INTENT(IN) when the actual argument is a PROTECTED_TARGET pointer or a subobject of one
  call noint_rank(ar)
  select rank (a => ar)
  rank (1)
    q => a ! F'2028 draft 11.1.12.3p3 gives POINTER, but not PROTECTED_TARGET.
  rank default
    !ERROR: POINTER dummy argument 'a=' must have the PROTECTED_TARGET attribute when the actual argument is a PROTECTED_TARGET pointer or a subobject of one
    call plainptrar_rank(a)
    !ERROR: Non-POINTER dummy argument 'a=' must have INTENT(IN) when the actual argument is a PROTECTED_TARGET pointer or a subobject of one
    call noint_rank(a)
    call ptar_rank(a)
    !ERROR: 'a' may not appear in NULLIFY
    !BECAUSE: 'ar' is a PROTECTED_TARGET pointer, so its target may not be defined through it
    nullify(a)
  end select
end

module c864_owner_module_target
  implicit none
  integer, target, protected :: x = 1
contains
  subroutine ok
    integer, pointer, protected_target :: p
    p => x
  end
end

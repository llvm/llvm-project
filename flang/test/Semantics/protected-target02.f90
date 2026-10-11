! RUN: %python %S/test_errors.py %s %flang_fc1
! Target definition is distinct from pointer association (8.5.16 p2).

module definitions
  type node
    integer :: value
    integer :: array(2)
    integer, pointer :: link
    integer, pointer, protected_target :: read_only_link
    integer, allocatable :: allocated
    procedure(), pointer, nopass :: proc
  end type
contains
  subroutine scalar(p)
    integer, pointer, protected_target :: p
    integer, target :: x
    p => x
    x = 1
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    p = 2
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    if (.false.) p = 3
    !ERROR: Input variable 'p' is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    read *, p
    !ERROR: 'p' may not be used as a DO variable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    do p = 1, 2
    end do
    associate(a => p)
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
      a = 2
    end associate
    nullify(p)
  end subroutine

  subroutine arrays(p, c, z)
    integer, pointer, protected_target :: p(:)
    character(:), pointer, protected_target :: c
    complex, pointer, protected_target :: z
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    p(1:2) = 1
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    p([1,2]) = 2
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'c' is not definable via that pointer
    c(1:1) = 'a'
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'z' is not definable via that pointer
    z%re = 1.
  end subroutine

  subroutine components(p, ordinary)
    type(node), pointer, protected_target :: p
    type(node), pointer :: ordinary
    integer, target :: x
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    p%value = 1
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    p%array(1) = 1
    ! A pointer component's separate target is not a subobject of p's target.
    p%link = 1
    !ERROR: The left-hand side of a pointer assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    p%link => x
    !ERROR: 'link' may not appear in NULLIFY
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    nullify(p%link)
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'read_only_link' is not definable via that pointer
    p%read_only_link = 1
    !ERROR: Name in ALLOCATE statement is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    allocate(p%allocated)
    !ERROR: Name in DEALLOCATE statement is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    deallocate(p%link)
    !ERROR: The left-hand side of a pointer assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    p%proc => target_proc
    ordinary%value = 1
    ordinary%link => x
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'read_only_link' is not definable via that pointer
    ordinary%read_only_link = 1
    ordinary%read_only_link => x
    associate(a => p)
      !ERROR: The left-hand side of a pointer assignment is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
      a%link => x
    end associate
  end subroutine

  subroutine select_rank_definition(p)
    integer, pointer, protected_target :: p(..)
    select rank(p)
    rank(0)
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
      p = 1
      nullify(p)
    rank(1)
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
      p(1) = 1
    end select
  end subroutine

  subroutine select_type_definition(p)
    class(*), pointer, protected_target :: p
    select type(a => p)
    type is(integer)
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
      a = 1
    end select
  end subroutine

  subroutine status_variables(p, msg)
    integer, pointer, protected_target :: p
    character(:), pointer, protected_target :: msg
    integer, allocatable :: a
    !ERROR: STAT variable 'p' is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    allocate(a, stat=p)
    !ERROR: ERRMSG variable 'msg' is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'msg' is not definable via that pointer
    deallocate(a, errmsg=msg)
    !ERROR: IOSTAT variable 'p' is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    write(*,*,iostat=p) 1
    !ERROR: Internal file variable 'msg' is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'msg' is not definable via that pointer
    write(msg,*) 1
  end subroutine

  subroutine target_proc
  end subroutine
end module

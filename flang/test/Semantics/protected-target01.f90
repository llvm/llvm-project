! RUN: %python %S/test_errors.py %s %flang_fc1
! PROTECTED_TARGET attribute (F2028 draft J3/26-007r1): declarations (C781,
! C867, C8109) and definitions of the target (C866, C868, C869, 8.5.16p2)
module m
  real, target :: x0
  real, pointer, protected_target :: p0
  !ERROR: A PROTECTED_TARGET entity must be a data pointer
  real, protected_target :: notPointer
  type t
    real, pointer, protected_target :: c1
    !ERROR: A PROTECTED_TARGET entity must be a data pointer
    real, protected_target :: c2
    real :: v
    real, pointer :: q
    real, allocatable :: a(:)
  end type
 contains
  function getp() result(r)
    real, pointer, protected_target :: r
    r => x0
  end
  subroutine statements
    real, pointer :: sp
    !ERROR: A PROTECTED_TARGET entity must be a data pointer
    real :: notPointer2
    !ERROR: A PROTECTED_TARGET entity must be a data pointer
    procedure(), pointer :: procPtr
    protected_target :: sp, notPointer2
    protected_target procPtr
  end
  subroutine commonBlock
    !ERROR: A PROTECTED_TARGET entity may not be in a common block
    real, pointer, protected_target :: pc
    common /blk/ pc
  end
  subroutine definitions(p, x, n)
    real, pointer, protected_target :: p(:)
    real, target :: x(:)
    integer, pointer, protected_target :: n
    type(t), pointer, protected_target :: pt
    type(t), target :: tt
    p => x
    nullify(p)
    p => x
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    p = 1.
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    p(1:1) = 1.
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    if (.false.) p(1) = 2.
    !ERROR: Input variable 'p' is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    read *, p
    !ERROR: 'n' may not be used as a DO variable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'n' is not definable via that pointer
    do n = 1, 2
    end do
    x = 1. ! the target is still definable by other means
    pt => tt
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'pt' is not definable via that pointer
    pt%v = 1.
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'pt' is not definable via that pointer
    pt%a = [1.]
    !ERROR: The left-hand side of a pointer assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'pt' is not definable via that pointer
    pt%q => x(1)
    !ERROR: 'q' may not appear in NULLIFY
    !BECAUSE: The target of PROTECTED_TARGET pointer 'pt' is not definable via that pointer
    nullify(pt%q)
    pt%q = 1. ! the target of a pointer component is not the target of pt
    tt%c1 => x(1)
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'c1' is not definable via that pointer
    tt%c1 = 1.
    associate (a => p, b => p(1), c => pt%q)
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
      a = 1.
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
      b = 1.
      c = 1.
    end associate
    print *, p, n, pt%v, tt%c1, getp()
  end
  subroutine selectors(u, rr)
    class(*), pointer, protected_target :: u
    real, pointer, protected_target :: rr(..)
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'r' is not definable via that pointer
    getp() = 1.
    associate (d => getp())
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'r' is not definable via that pointer
      d = 2.
    end associate
    select type (v => u)
    type is (integer)
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'u' is not definable via that pointer
      v = 1
    end select
    select rank (a => rr)
    rank (1)
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'rr' is not definable via that pointer
      a(1) = 1.
    end select
  end
  subroutine allocations
    real, pointer, protected_target :: p, ps(:)
    type(t), pointer, protected_target :: pt
    type(t) :: tt
    !ERROR: SOURCE= must appear in ALLOCATE of PROTECTED_TARGET pointer 'p'
    allocate(p)
    !ERROR: SOURCE= must appear in ALLOCATE of PROTECTED_TARGET pointer 'ps'
    allocate(ps(2), mold=[1., 2.])
    !ERROR: SOURCE= must appear in ALLOCATE of PROTECTED_TARGET pointer 'c1'
    allocate(tt%c1)
    allocate(p, source=1.)
    allocate(ps, source=[1., 2.])
    allocate(tt%c1, source=2.)
    allocate(pt, source=tt)
    !ERROR: Name in ALLOCATE statement is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'pt' is not definable via that pointer
    allocate(pt%a(2))
    !ERROR: Object in DEALLOCATE statement is not deallocatable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    deallocate(p)
    !ERROR: Object in DEALLOCATE statement is not deallocatable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'c1' is not definable via that pointer
    deallocate(tt%c1)
    !ERROR: Name in DEALLOCATE statement is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'pt' is not definable via that pointer
    deallocate(pt%a)
    nullify(p, ps, pt)
  end
end

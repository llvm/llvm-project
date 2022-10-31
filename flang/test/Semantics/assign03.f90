! RUN: %python %S/test_errors.py %s %flang_fc1
! Pointer assignment constraints 10.2.2.2 (see also assign02.f90)

module m
  interface
    subroutine s(i)
      integer i
    end
  end interface
  type :: t
    procedure(s), pointer, nopass :: p
    real, pointer :: q
  end type
contains
  ! C1027
  subroutine s1
    type(t), allocatable :: a(:)
    type(t), allocatable :: b[:]
    a(1)%p => s
    !ERROR: Procedure pointer may not be a coindexed object
    b[1]%p => s
  end
  ! C1028
  subroutine s2
    type(t) :: a
    a%p => s
    !ERROR: In assignment to object pointer 'q', the target 's' is a procedure designator
    a%q => s
  end
  ! C1029
  subroutine s3
    type(t) :: a
    a%p => f()  ! OK: pointer-valued function
    !ERROR: Subroutine pointer 'p' may not be associated with function designator 'f'
    a%p => f
  contains
    function f()
      procedure(s), pointer :: f
      f => s
    end
  end

  ! C1030 and 10.2.2.4 - procedure names as target of procedure pointer
  subroutine s4(s_dummy)
    procedure(s) :: s_dummy
    procedure(s), pointer :: p, q
    procedure(), pointer :: r
    integer :: i
    external :: s_external
    p => s_dummy
    p => s_internal
    p => s_module
    q => p
    r => s_external
  contains
    subroutine s_internal(i)
      integer i
    end
  end
  subroutine s_module(i)
    integer i
  end

  ! 10.2.2.4(3)
  subroutine s5
    procedure(f_impure1), pointer :: p_impure
    procedure(f_pure1), pointer :: p_pure
    !ERROR: Procedure pointer 'p_elemental' may not be ELEMENTAL
    procedure(f_elemental1), pointer :: p_elemental
    procedure(s_impure1), pointer :: sp_impure
    procedure(s_pure1), pointer :: sp_pure
    !ERROR: Procedure pointer 'sp_elemental' may not be ELEMENTAL
    procedure(s_elemental1), pointer :: sp_elemental

    p_impure => f_impure1 ! OK, same characteristics
    p_impure => f_pure1 ! OK, target may be pure when pointer is not
    !ERROR: Procedure pointer 'p_impure' associated with incompatible procedure designator 'f_elemental1': incompatible procedure attributes: Elemental
    p_impure => f_elemental1
    !ERROR: Procedure pointer 'p_impure' associated with incompatible procedure designator 'f_impureelemental1': incompatible procedure attributes: Elemental
    p_impure => f_ImpureElemental1 ! OK, target may be elemental

    sp_impure => s_impure1 ! OK, same characteristics
    sp_impure => s_pure1 ! OK, target may be pure when pointer is not
    !ERROR: Procedure pointer 'sp_impure' associated with incompatible procedure designator 's_elemental1': incompatible procedure attributes: Elemental
    sp_impure => s_elemental1

    !ERROR: PURE procedure pointer 'p_pure' may not be associated with non-PURE procedure designator 'f_impure1'
    p_pure => f_impure1
    p_pure => f_pure1 ! OK, same characteristics
    !ERROR: Procedure pointer 'p_pure' associated with incompatible procedure designator 'f_elemental1': incompatible procedure attributes: Elemental
    p_pure => f_elemental1
    !ERROR: PURE procedure pointer 'p_pure' may not be associated with non-PURE procedure designator 'f_impureelemental1'
    p_pure => f_impureElemental1

    !ERROR: PURE procedure pointer 'sp_pure' may not be associated with non-PURE procedure designator 's_impure1'
    sp_pure => s_impure1
    sp_pure => s_pure1 ! OK, same characteristics
    !ERROR: Procedure pointer 'sp_pure' associated with incompatible procedure designator 's_elemental1': incompatible procedure attributes: Elemental
    sp_pure => s_elemental1 ! OK, target may be elemental when pointer is not

    !ERROR: Procedure pointer 'p_impure' associated with incompatible procedure designator 'f_impure2': incompatible dummy argument #1: incompatible dummy data object intents
    p_impure => f_impure2
    !ERROR: Procedure pointer 'p_pure' associated with incompatible procedure designator 'f_pure2': function results have incompatible types: INTEGER(4) vs REAL(4)
    p_pure => f_pure2
    !ERROR: Procedure pointer 'p_impure' associated with incompatible procedure designator 'f_elemental2': incompatible procedure attributes: Elemental
    p_impure => f_elemental2

    !ERROR: Procedure pointer 'sp_impure' associated with incompatible procedure designator 's_impure2': incompatible procedure attributes: BindC
    sp_impure => s_impure2
    !ERROR: Procedure pointer 'sp_impure' associated with incompatible procedure designator 's_pure2': incompatible dummy argument #1: incompatible dummy data object intents
    sp_impure => s_pure2
    !ERROR: Procedure pointer 'sp_pure' associated with incompatible procedure designator 's_elemental2': incompatible procedure attributes: Elemental
    sp_pure => s_elemental2

    !ERROR: Function pointer 'p_impure' may not be associated with subroutine designator 's_impure1'
    p_impure => s_impure1

    !ERROR: Subroutine pointer 'sp_impure' may not be associated with function designator 'f_impure1'
    sp_impure => f_impure1

  contains
    integer function f_impure1(n)
      real, intent(in) :: n
      f_impure = n
    end
    pure integer function f_pure1(n)
      real, intent(in) :: n
      f_pure = n
    end
    elemental integer function f_elemental1(n)
      real, intent(in) :: n
      f_elemental = n
    end
    impure elemental integer function f_impureElemental1(n)
      real, intent(in) :: n
      f_impureElemental = n
    end

    integer function f_impure2(n)
      real, intent(inout) :: n
      f_impure = n
    end
    pure real function f_pure2(n)
      real, intent(in) :: n
      f_pure = n
    end
    elemental integer function f_elemental2(n)
      real, value :: n
      f_elemental = n
    end

    subroutine s_impure1(n)
      integer, intent(inout) :: n
      n = n + 1
    end
    pure subroutine s_pure1(n)
      integer, intent(inout) :: n
      n = n + 1
    end
    elemental subroutine s_elemental1(n)
      integer, intent(inout) :: n
      n = n + 1
    end

    subroutine s_impure2(n) bind(c)
      integer, intent(inout) :: n
      n = n + 1
    end subroutine s_impure2
    pure subroutine s_pure2(n)
      integer, intent(out) :: n
      n = 1
    end subroutine s_pure2
    elemental subroutine s_elemental2(m,n)
      integer, intent(inout) :: m, n
      n = m + n
    end subroutine s_elemental2
  end

  ! 10.2.2.4(4)
  subroutine s6
    procedure(s), pointer :: p, q
    procedure(), pointer :: r
    external :: s_external
    p => s_external ! OK for a pointer with an explicit interface to be associated with a procedure with an implicit interface
    r => s_module ! OK for a pointer with implicit interface to be associated with a procedure with an explicit interface.  See 10.2.2.4 (3)
  end

  ! 10.2.2.4(5)
  subroutine s7
    procedure(real) :: f_external
    external :: s_external
    procedure(), pointer :: p_s
    procedure(real), pointer :: p_f
    p_f => f_external
    p_s => s_external
    !Ok: p_s has no interface
    p_s => f_external
    !Ok: s_external has no interface
    p_f => s_external
  end

  ! C1017: bounds-spec
  subroutine s8
    real, target :: x(10, 10)
    real, pointer :: p(:, :)
    p(2:,3:) => x
    !ERROR: Pointer 'p' has rank 2 but the number of bounds specified is 1
    p(2:) => x
  end

  ! bounds-remapping
  subroutine s9
    real, target :: x(10, 10), y(100)
    real, pointer :: p(:, :)
    ! C1018
    !ERROR: Pointer 'p' has rank 2 but the number of bounds specified is 1
    p(1:100) => x
    ! 10.2.2.3(9)
    !ERROR: Pointer bounds remapping target must have rank 1 or be simply contiguous
    p(1:5,1:5) => x(1:10,::2)
    ! 10.2.2.3(9)
    !ERROR: Pointer bounds require 25 elements but target has only 20
    p(1:5,1:5) => x(:,1:2)
    !OK - rhs has rank 1 and enough elements
    p(1:5,1:5) => y(1:100:2)
    !OK - same, but from function result
    p(1:5,1:5) => f()
   contains
    function f()
      real, pointer :: f(:)
      f => y
    end function
  end

  subroutine s10
    integer, pointer :: p(:)
    type :: t
      integer :: a(4, 4)
      integer :: b
    end type
    type(t), target :: x
    type(t), target :: y(10,10)
    integer :: v(10)
    p(1:16) => x%a
    p(1:8) => x%a(:,3:4)
    p(1:1) => x%b  ! We treat scalars as simply contiguous
    p(1:1) => x%a(1,1)
    p(1:1) => y(1,1)%a(1,1)
    p(1:1) => y(:,1)%a(1,1)  ! Rank 1 RHS
    !ERROR: Pointer bounds remapping target must have rank 1 or be simply contiguous
    p(1:4) => x%a(::2,::2)
    !ERROR: Pointer bounds remapping target must have rank 1 or be simply contiguous
    p(1:100) => y(:,:)%b
    !ERROR: Pointer bounds remapping target must have rank 1 or be simply contiguous
    p(1:100) => y(:,:)%a(1,1)
    !ERROR: Pointer bounds remapping target must have rank 1 or be simply contiguous
    !ERROR: An array section with a vector subscript may not be a pointer target
    p(1:4) => x%a(:,v)
  end

  subroutine s11
    complex, target :: x(10,10)
    complex, pointer :: p(:)
    real, pointer :: q(:)
    p(1:100) => x(:,:)
    q(1:10) => x(1,:)%im
    !ERROR: Pointer bounds remapping target must have rank 1 or be simply contiguous
    q(1:100) => x(:,:)%re
  end

  ! Check is_contiguous, which is usually the same as when pointer bounds
  ! remapping is used.
  subroutine s12
    integer, pointer :: p(:)
    integer, pointer, contiguous :: pc(:)
    type :: t
      integer :: a(4, 4)
      integer :: b
    end type
    type(t), target :: x
    type(t), target :: y(10,10)
    integer :: v(10)
    logical(kind=merge(1,-1,is_contiguous(x%a(:,:)))) :: l1 ! known true
    logical(kind=merge(1,-1,is_contiguous(y(1,1)%a(1,1)))) :: l2 ! known true
    !ERROR: Must be a constant value
    logical(kind=merge(-1,-2,is_contiguous(y(:,1)%a(1,1)))) :: l3 ! unknown
    !ERROR: Must be a constant value
    logical(kind=merge(-1,-2,is_contiguous(y(:,1)%a(1,1)))) :: l4 ! unknown
    logical(kind=merge(-1,1,is_contiguous(x%a(:,v)))) :: l5 ! known false
    !ERROR: Must be a constant value
    logical(kind=merge(-1,-2,is_contiguous(y(v,1)%a(1,1)))) :: l6 ! unknown
    !ERROR: Must be a constant value
    logical(kind=merge(-1,-2,is_contiguous(p(:)))) :: l7 ! unknown
    logical(kind=merge(1,-1,is_contiguous(pc(:)))) :: l8 ! known true
    logical(kind=merge(-1,1,is_contiguous(pc(1:10:2)))) :: l9 ! known false
    logical(kind=merge(-1,1,is_contiguous(pc(10:1:-1)))) :: l10 ! known false
    logical(kind=merge(1,-1,is_contiguous(pc(1:10:1)))) :: l11 ! known true
    logical(kind=merge(-1,1,is_contiguous(pc(10:1:-1)))) :: l12 ! known false
    !ERROR: Must be a constant value
    logical(kind=merge(-1,1,is_contiguous(pc(::-1)))) :: l13 ! unknown (could be empty)
    logical(kind=merge(1,-1,is_contiguous(y(1,1)%a(::-1,1)))) :: l14 ! known true (empty)
    logical(kind=merge(1,-1,is_contiguous(y(1,1)%a(1,::-1)))) :: l15 ! known true (empty)
  end
  subroutine test3(b)
    integer, intent(inout) :: b(..)
    !ERROR: Must be a constant value
    integer, parameter :: i = rank(b)
  end subroutine

  subroutine s13
    external :: s_external
    procedure(), pointer :: ptr
    !Ok - don't emit an error about incompatible Subroutine attribute
    ptr => s_external
    call ptr
  end subroutine

  subroutine s14
    procedure(real), pointer :: ptr
    sf(x) = x + 1.
    !ERROR: Statement function 'sf' may not be the target of a pointer assignment
    ptr => sf
  end subroutine
end

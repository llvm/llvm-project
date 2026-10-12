! RUN: %python %S/test_errors.py %s %flang_fc1
! 9.4.5
subroutine s1
  type :: t(k, l)
    integer, kind :: k
    integer, len :: l
  end type
  type(t(1, 2)) :: x
  !ERROR: Assignment to constant 'x%k' is not allowed
  x%k = 4
  !ERROR: Assignment to constant 'x%l' is not allowed
  x%l = 3
end

! C901
subroutine s2(x)
  !ERROR: A dummy argument may not also be a named constant
  real, parameter :: x = 0.0
  real, parameter :: a(*) = [1, 2, 3]
  character, parameter :: c(2) = "ab"
  integer :: i
  !ERROR: Assignment to constant 'x' is not allowed
  x = 2.0
  i = 2
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'a' is not a variable
  a(i) = 3.0
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'a' is not a variable
  a(i:i+1) = [4, 5]
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'c' is not a variable
  c(i:2) = "cd"
end

! C901
subroutine s3
  type :: t
    integer :: a(2)
    integer :: b
  end type
  type(t) :: x
  type(t), parameter :: y = t([1,2], 3)
  integer :: i = 1
  x%a(i) = 1
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'y' is not a variable
  y%a(i) = 2
  x%b = 4
  !ERROR: Assignment to constant 'y%b' is not allowed
  y%b = 5
end

! C844
subroutine s4
  type :: t
    integer :: a(2)
  end type
contains
  subroutine s(x, c)
    type(t), intent(in) :: x
    character(10), intent(in) :: c
    type(t) :: y
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'x' is an INTENT(IN) dummy argument
    x = y
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'x' is an INTENT(IN) dummy argument
    x%a(1) = 2
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'c' is an INTENT(IN) dummy argument
    c(2:3) = "ab"
  end
end

! 8.5.15(2)
module m5
  real :: x
  real, protected :: y
  real, private :: z
  type :: t
    real :: a
  end type
  type(t), protected :: b
end
subroutine s5()
  use m5
  implicit none
  x = 1.0
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'y' is protected in this scope
  y = 2.0
  !ERROR: No explicit type declared for 'z'
  z = 3.0
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'b' is protected in this scope
  b%a = 1.0
end

subroutine s6(x)
  integer :: x(*)
  integer, allocatable :: ja(:)
  x(1:3) = [1, 2, 3]
  x(:3) = [1, 2, 3]
  !ERROR: Assumed-size array 'x' must have explicit final subscript upper bound value
  x(:) = [1, 2, 3]
  !ERROR: Whole assumed-size array 'x' may not appear here without subscripts
  x = [1, 2, 3]
  !ERROR: Whole assumed-size array 'x' may not appear here without subscripts
  ja = x
  associate (y => x) ! ok
    !ERROR: Whole assumed-size array 'y' may not appear here without subscripts
    y = [1, 2, 3]
  end associate
  !ERROR: Whole assumed-size array 'x' may not appear here without subscripts
  associate (y => (x))
  end associate
end

module m7
  type :: t
    integer :: i
  end type
contains
  subroutine s7(x)
    type(t) :: x(*)
    x(:3)%i = [1, 2, 3]
    !ERROR: Whole assumed-size array 'x' may not appear here without subscripts
    x%i = [1, 2, 3]
  end
end

subroutine s7
  integer :: a(10), v(10)
  a(v(:)) = 1  ! vector subscript is ok
end

subroutine s8
  !ERROR: Assignment to procedure 's8' is not allowed
  s8 = 1.0
end

real function f9() result(r)
  !ERROR: Assignment to procedure 'f9' is not allowed
  f9 = 1.0
end

subroutine s9
  real f9a
  !ERROR: Assignment to procedure 'f9a' is not allowed
  f9a = 1.0
  print *, f9a(1)
end

!ERROR: No explicit type declared for dummy argument 'n'
subroutine s10(a, n)
  implicit none
  real a(n)
  a(1:n) = 0.0  ! should not get a second error here
end

subroutine s11
  intrinsic :: sin
  real :: a
  !ERROR: Function call must have argument list
  a = sin
  !ERROR: Subroutine name is not allowed here
  a = s11
end

subroutine s12()
  type dType(l1, k1, l2, k2)
    integer, len :: l1
    integer, kind :: k1
    integer, len :: l2
    integer, kind :: k2
  end type

  contains
    subroutine sub(arg1, arg2, arg3)
      integer :: arg1
      type(dType(arg1, 2, *, 4)) :: arg2
      type(dType(*, 2, arg1, 4)) :: arg3
      type(dType(1, 2, 3, 4)) :: local1
      type(dType(1, 2, 3, 4)) :: local2
      type(dType(1, 2, arg1, 4)) :: local3
      type(dType(9, 2, 3, 4)) :: local4
      type(dType(1, 9, 3, 4)) :: local5

      arg2 = arg3
      arg2 = local1
      arg3 = local1
      local1 = local2
      local2 = local3
      !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types TYPE(dtype(k1=2_4,k2=4_4,l1=1_4,l2=3_4)) and TYPE(dtype(k1=2_4,k2=4_4,l1=9_4,l2=3_4))
      local1 = local4 ! mismatched constant KIND type parameter
      !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types TYPE(dtype(k1=2_4,k2=4_4,l1=1_4,l2=3_4)) and TYPE(dtype(k1=9_4,k2=4_4,l1=1_4,l2=3_4))
      local1 = local5 ! mismatched constant LEN type parameter
    end subroutine sub
end subroutine s12

subroutine s13()
  interface assignment(=)
    procedure :: cToR, cToRa, cToI
  end interface
  real :: x(1)
  integer :: n(1)
  x='0' ! fine
  n='0' ! fine
  !ERROR: Defined assignment in WHERE must be elemental, but 'ctora' is not
  where ([1==1]) x='*'
  where ([1==1]) n='*' ! fine
  forall (j=1:1)
    !ERROR: The mask or variable must not be scalar
    where (j==1)
      !ERROR: Defined assignment in WHERE must be elemental, but 'ctor' is not
      !ERROR: The mask or variable must not be scalar
      x(j)='?'
      !ERROR: The mask or variable must not be scalar
      n(j)='?'
    !ERROR: The mask or variable must not be scalar
    elsewhere (.false.)
      !ERROR: Defined assignment in WHERE must be elemental, but 'ctor' is not
      !ERROR: The mask or variable must not be scalar
      x(j)='1'
      !ERROR: The mask or variable must not be scalar
      n(j)='1'
    elsewhere
      !ERROR: Defined assignment in WHERE must be elemental, but 'ctor' is not
      !ERROR: The mask or variable must not be scalar
      x(j)='9'
      !ERROR: The mask or variable must not be scalar
      n(j)='9'
    end where
  end forall
  x='0' ! still fine
  n='0' ! still fine
 contains
  subroutine cToR(x, c)
    real, intent(out) :: x
    character, intent(in) :: c
  end subroutine
  subroutine cToRa(x, c)
    real, intent(out) :: x(:)
    character, intent(in) :: c
  end subroutine
  elemental subroutine cToI(n, c)
    integer, intent(out) :: n
    character, intent(in) :: c
  end subroutine
end subroutine s13

module m14
  type t1
    integer, pointer :: p
   contains
    procedure definedAsst1
    generic :: assignment(=) => definedAsst1
  end type
  type t2
    integer, pointer :: p
  end type
  interface assignment(=)
    module procedure definedAsst2
  end interface
  type t3
    integer, pointer :: p
  end type
 contains
  pure subroutine definedAsst1(lhs,rhs)
    class(t1), intent(in out) :: lhs
    class(t1), intent(in) :: rhs
  end subroutine
  pure subroutine definedAsst2(lhs,rhs)
    type(t2), intent(out) :: lhs
    type(t2), intent(in) :: rhs
  end subroutine
  pure subroutine test(y1,y2,y3)
    type(t1) x1
    type(t1), intent(in) :: y1
    type(t2) x2
    type(t2), intent(in) :: y2
    type(t3) x3
    type(t3), intent(in) :: y3
    x1 = y1 ! fine due to not being intrinsic assignment
    x2 = y2 ! fine due to not being intrinsic assignment
    !ERROR: A pure subprogram may not copy the value of 'y3' because it is an INTENT(IN) dummy argument and has the POINTER potential subobject component '%p'
    x3 = y3
  end subroutine
end module m14

module m15_protected_target
  implicit none
  type :: bad_component
    !ERROR: A PROTECTED_TARGET entity must be a data pointer
    integer, protected_target :: x
  end type
  type :: t870
    integer, pointer, protected_target :: p
  end type
  type :: t871_drop
    integer, pointer :: p
  end type
  type :: t871_keep
    integer, pointer, protected_target :: p
  end type
  type :: node
    integer, pointer :: next
  end type
  type :: leaf
    integer :: value
  end type
  type :: linked_node
    type(linked_node), pointer :: next
    integer :: value
  end type
  type :: alloc_node
    integer, allocatable :: values(:)
    type(linked_node), pointer :: next
  end type
  type :: proc_node
    procedure(), pointer, nopass :: fun
  end type
  type :: alloc_holder
    type(alloc_node), pointer, protected_target :: p
  end type
  type :: linked_holder
    type(linked_node), pointer, protected_target :: p
  end type
  type :: proc_holder
    type(proc_node), pointer, protected_target :: p
  end type
  type :: pt_node
    integer, pointer, protected_target :: child
    type(pt_node), pointer :: next
    integer :: value
  end type
  type :: pt_holder
    type(pt_node), pointer, protected_target :: p
    type(pt_node), pointer :: q
  end type
  type :: protected_box
    type(linked_node), pointer :: child
    type(leaf) :: fixed
    integer :: values(4)
  end type
  type(protected_box), protected, target :: owned_box
  integer, protected, target :: owned_x
  abstract interface
    function protected_result() result(p)
      integer, pointer, protected_target :: p
    end function
  end interface
contains
  subroutine c8109_protected_target_stmt()
    !ERROR: A PROTECTED_TARGET entity must be a data pointer
    real :: x
    protected_target :: x
  end subroutine

  subroutine protected_target_statement_forms()
    integer, target :: x
    integer, pointer :: c, d
    protected_target c
    protected_target :: d
    c => x
    d => c
  end subroutine

  subroutine c867_protected_target_common()
    !ERROR: A pointer with the PROTECTED_TARGET attribute shall not be in a common block
    integer, pointer, protected_target :: p
    common /m15_blk/ p
  end subroutine

  subroutine c866_scalar_write()
    integer, target :: x
    integer, pointer, protected_target :: p
    p => x
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    p = 1
  end subroutine

  subroutine c866_section_write()
    integer, target :: x(4)
    integer, pointer, protected_target :: p(:)
    p => x
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    p(2:3) = 1
  end subroutine

  subroutine c866_subcomponent_write()
    type(leaf), target :: x
    type(leaf), pointer, protected_target :: p
    p => x
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    p%value = 1
  end subroutine

  subroutine c866_substring_write()
    character(4), target :: x
    character(:), pointer, protected_target :: p
    p => x
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    p(2:3) = 'xx'
  end subroutine

  subroutine c866_associate_section_write()
    integer, target :: x(4)
    integer, pointer, protected_target :: p(:)
    p => x
    associate (a => p(2:))
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
      a = 1
      associate (b => a)
        !ERROR: Left-hand side of assignment is not definable
        !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
        b = 1
      end associate
    end associate
  end subroutine

  subroutine c868_protected_target_deallocate()
    integer, pointer, protected_target :: p
    !ERROR: A PROTECTED_TARGET pointer may not be deallocated
    deallocate(p)
  end subroutine

  subroutine c869_protected_target_allocate()
    integer, target :: x
    integer, pointer, protected_target :: p
    p => x
    !ERROR: ALLOCATE of object associated with PROTECTED_TARGET pointer 'p' requires SOURCE=
    allocate(p)
    allocate(p, source=x)
  end subroutine

  subroutine independent_pointer_target_write_ok(p)
    type(node), pointer, protected_target :: p
    p%next = 1
  end subroutine

  subroutine pointer_component_reassociation_reject(p, x)
    type(node), pointer, protected_target :: p
    integer, target :: x
    !ERROR: The left-hand side of a pointer assignment is not definable
    !BECAUSE: A pointer subobject of the target of PROTECTED_TARGET pointer 'p' is not definable
    p%next => x
  end subroutine

  subroutine allocate_pointer_component_reject(p)
    type(node), pointer, protected_target :: p
    !ERROR: Name in ALLOCATE statement is not definable
    !BECAUSE: A pointer subobject of the target of PROTECTED_TARGET pointer 'p' is not definable
    allocate(p%next, source=1)
  end subroutine

  subroutine nested_allocatable_component_reject(h)
    type(alloc_holder) :: h
    !ERROR: Name in ALLOCATE statement is not definable
    !BECAUSE: An allocatable subobject of the target of PROTECTED_TARGET pointer 'p' is not definable
    allocate(h%p%values(1), source=1)
  end subroutine

  subroutine nested_pointer_component_reassociation_reject(h, x)
    type(linked_holder) :: h
    type(linked_node), target :: x
    !ERROR: The left-hand side of a pointer assignment is not definable
    !BECAUSE: A pointer subobject of the target of PROTECTED_TARGET pointer 'p' is not definable
    h%p%next => x
  end subroutine

  subroutine nested_pointer_component_nullify_reject(h)
    type(linked_holder) :: h
    !ERROR: 'next' may not appear in NULLIFY
    !BECAUSE: A pointer subobject of the target of PROTECTED_TARGET pointer 'p' is not definable
    nullify(h%p%next)
  end subroutine

  subroutine nested_pointer_component_allocate_reject(h)
    type(linked_holder) :: h
    !ERROR: Name in ALLOCATE statement is not definable
    !BECAUSE: A pointer subobject of the target of PROTECTED_TARGET pointer 'p' is not definable
    allocate(h%p%next, source=linked_node(null(), 1))
  end subroutine

  subroutine nested_pointer_component_deallocate_reject(h)
    type(linked_holder) :: h
    !ERROR: An object associated with PROTECTED_TARGET pointer 'p' may not be deallocated
    deallocate(h%p%next)
  end subroutine

  subroutine independent_pointer_descriptor_ok(p, x)
    type(linked_node), pointer, protected_target :: p
    type(linked_node), target :: x
    p%next%next => x
    nullify(p%next%next)
    allocate(p%next%next, source=linked_node(null(), 1))
  end subroutine

  subroutine protected_component_descriptors(p, h, q, x)
    type(pt_node), pointer, protected_target :: p
    type(pt_holder) :: h
    type(pt_node), pointer :: q
    integer, target :: x
    p => q
    nullify(p)
    allocate(p, source=pt_node(null(), null(), 1))
    q%child => x
    nullify(q%child)
    allocate(q%child, source=1)
    call takes_pointer_pt_out(q%child)
    call takes_pointer_pt_inout(q%child)
    call takes_pointer_pt_default(q%child)
    call takes_pointer_pt(p%child)
    !ERROR: The left-hand side of a pointer assignment is not definable
    !BECAUSE: A pointer subobject of the target of PROTECTED_TARGET pointer 'p' is not definable
    p%child => x
    !ERROR: 'child' may not appear in NULLIFY
    !BECAUSE: A pointer subobject of the target of PROTECTED_TARGET pointer 'p' is not definable
    nullify(p%child)
    !ERROR: Name in ALLOCATE statement is not definable
    !BECAUSE: A pointer subobject of the target of PROTECTED_TARGET pointer 'p' is not definable
    allocate(p%child, source=1)
    !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'a=' is not definable
    !BECAUSE: A pointer subobject of the target of PROTECTED_TARGET pointer 'p' is not definable
    call takes_pointer_pt_out(p%child)
    !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'a=' is not definable
    !BECAUSE: A pointer subobject of the target of PROTECTED_TARGET pointer 'p' is not definable
    call takes_pointer_pt_inout(p%child)
    !ERROR: The left-hand side of a pointer assignment is not definable
    !BECAUSE: A pointer subobject of the target of PROTECTED_TARGET pointer 'p' is not definable
    h%p%child => x
    !ERROR: 'child' may not appear in NULLIFY
    !BECAUSE: A pointer subobject of the target of PROTECTED_TARGET pointer 'p' is not definable
    nullify(h%p%child)
    !ERROR: Name in ALLOCATE statement is not definable
    !BECAUSE: A pointer subobject of the target of PROTECTED_TARGET pointer 'p' is not definable
    allocate(h%p%child, source=1)
    !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'a=' is not definable
    !BECAUSE: A pointer subobject of the target of PROTECTED_TARGET pointer 'p' is not definable
    call takes_pointer_pt_out(h%p%child)
    h%q%child => x
    nullify(h%q%child)
    allocate(h%q%child, source=1)
    call takes_pointer_pt_out(h%q%child)
    p%next%child => x
    nullify(p%next%child)
    allocate(p%next%child, source=1)
    call takes_pointer_pt_out(p%next%child)
    p%next%value = 1
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'child' is not definable
    p%next%child = 1
    associate(a => p)
      !ERROR: The left-hand side of a pointer assignment is not definable
      !BECAUSE: A pointer subobject of the target of PROTECTED_TARGET pointer 'p' is not definable
      a%child => x
      associate(b => a)
        !ERROR: 'child' may not appear in NULLIFY
        !BECAUSE: A pointer subobject of the target of PROTECTED_TARGET pointer 'p' is not definable
        nullify(b%child)
      end associate
    end associate
    associate(a => p%next)
      a%child => x
      nullify(a%child)
      call takes_pointer_pt_out(a%child)
      a%value = 1
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'child' is not definable
      a%child = 1
    end associate
  end subroutine

  subroutine procedure_component_reject(p, f)
    type(proc_node), pointer, protected_target :: p
    procedure() :: f
    !ERROR: The left-hand side of a pointer assignment is not definable
    !BECAUSE: A pointer subobject of the target of PROTECTED_TARGET pointer 'p' is not definable
    p%fun => f
    !ERROR: 'fun' may not appear in NULLIFY
    !BECAUSE: A pointer subobject of the target of PROTECTED_TARGET pointer 'p' is not definable
    nullify(p%fun)
  end subroutine

  subroutine nested_procedure_component_reject(h, f)
    type(proc_holder) :: h
    procedure() :: f
    !ERROR: The left-hand side of a pointer assignment is not definable
    !BECAUSE: A pointer subobject of the target of PROTECTED_TARGET pointer 'p' is not definable
    h%p%fun => f
    !ERROR: 'fun' may not appear in NULLIFY
    !BECAUSE: A pointer subobject of the target of PROTECTED_TARGET pointer 'p' is not definable
    nullify(h%p%fun)
  end subroutine

  subroutine c870_drop_protection(x)
    integer, target :: x
    integer, pointer, protected_target :: p
    integer, pointer :: q
    p => x
    !ERROR: Pointer assignment target 'p' has the PROTECTED_TARGET attribute; pointer 'q' must also have the PROTECTED_TARGET attribute
    q => p
  end subroutine

  subroutine c870_subobject_drop(x)
    integer, target :: x
    type(t870) :: t
    integer, pointer :: q
    t%p => x
    !ERROR: Pointer assignment target 'p' has the PROTECTED_TARGET attribute; pointer 'q' must also have the PROTECTED_TARGET attribute
    q => t%p
  end subroutine

  subroutine c871_structure_constructor(targetObject)
    integer, target :: targetObject
    type(t871_drop) :: x
    type(t871_keep) :: y
    integer, pointer, protected_target :: p
    p => targetObject
    !ERROR: Structure constructor target 'p' has the PROTECTED_TARGET attribute; component 'p' must also have the PROTECTED_TARGET attribute
    x = t871_drop(p)
    y = t871_keep(p)
  end subroutine

  function protected_function_result() result(p)
    integer, pointer, protected_target :: p
    p => owned_x
  end function

  subroutine function_result_drop_keep()
    integer, pointer :: q
    integer, pointer, protected_target :: r
    !ERROR: Pointer assignment target is function 'protected_function_result' result with the PROTECTED_TARGET attribute; pointer 'q' must also have the PROTECTED_TARGET attribute
    q => protected_function_result()
    r => protected_function_result()
  end subroutine

  subroutine function_result_target_reject()
    integer, pointer :: q
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
    protected_function_result() = 1
    associate (a => protected_function_result())
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
      a = 1
      ! An associate name is not a pointer (F'2028 11.1.3.3), so C870-C874
      ! do not apply to it; only definitions through it are prohibited
      ! (8.5.16p2, C866 and 20.6.7(11)).
      q => a
      call takes_nonpointer_default(a)
    end associate
  end subroutine

  subroutine takes_pointer(a)
    integer, pointer, intent(in) :: a
  end subroutine

  subroutine takes_pointer_pt(a)
    integer, pointer, protected_target, intent(in) :: a
  end subroutine

  subroutine takes_pointer_pt_out(a)
    integer, pointer, protected_target, intent(out) :: a
  end subroutine

  subroutine takes_pointer_pt_inout(a)
    integer, pointer, protected_target, intent(inout) :: a
  end subroutine

  subroutine takes_pointer_pt_default(a)
    integer, pointer, protected_target :: a
  end subroutine

  subroutine takes_nonpointer_in(a)
    integer, intent(in) :: a
  end subroutine

  subroutine takes_nonpointer_default(a)
    integer :: a
  end subroutine

  subroutine takes_linked_pointer(a)
    type(linked_node), pointer, intent(in) :: a
  end subroutine

  subroutine takes_linked_pointer_default(a)
    type(linked_node), pointer :: a
  end subroutine

  subroutine takes_linked_pointer_pt(a)
    type(linked_node), pointer, protected_target, intent(in) :: a
  end subroutine

  subroutine takes_linked_in(a)
    type(linked_node), intent(in) :: a
  end subroutine

  subroutine takes_linked_default(a)
    type(linked_node) :: a
  end subroutine

  subroutine takes_proc_nonpointer(proc)
    procedure() :: proc
  end subroutine

  subroutine takes_proc_pointer_default(proc)
    procedure(), pointer :: proc
  end subroutine

  subroutine takes_proc_pointer_in(proc)
    procedure(), pointer, intent(in) :: proc
  end subroutine

  subroutine takes_proc_pointer_out(proc)
    procedure(), pointer, intent(out) :: proc
  end subroutine

  subroutine takes_proc_pointer_inout(proc)
    procedure(), pointer, intent(inout) :: proc
  end subroutine

  subroutine c872_c873_c874_calls()
    integer, target :: x
    integer, pointer :: q
    integer, pointer, protected_target :: p
    external :: implicit_sub
    p => x
    q => x
    call takes_pointer_pt(p)
    call takes_nonpointer_in(p)
    call takes_pointer(q)
    !ERROR: Actual argument with PROTECTED_TARGET pointer 'p' may be associated with pointer dummy argument 'a=' only if it also has the PROTECTED_TARGET attribute
    call takes_pointer(p)
    !ERROR: Actual argument with PROTECTED_TARGET pointer 'p' may be associated with non-pointer dummy argument 'a=' only if it has INTENT(IN)
    call takes_nonpointer_default(p)
    !ERROR: Actual argument with PROTECTED_TARGET pointer 'p' requires an explicit interface
    call implicit_sub(p)
  end subroutine

  subroutine procedure_dummy_result_call_drop(f)
    procedure(protected_result) :: f
    !ERROR: Actual argument with PROTECTED_TARGET pointer 'p' may be associated with pointer dummy argument 'a=' only if it also has the PROTECTED_TARGET attribute
    call takes_pointer(f())
    call takes_pointer_pt(f())
  end subroutine

  subroutine c872_component_calls(p)
    type(linked_node), pointer, protected_target :: p
    call takes_linked_pointer_pt(p%next)
    call takes_linked_in(p%next)
    call takes_linked_default(p%next)
    call takes_linked_pointer(p%next%next)
    call takes_linked_pointer_default(p%next%next)
    !ERROR: Actual argument with PROTECTED_TARGET pointer 'p' may be associated with pointer dummy argument 'a=' only if it also has the PROTECTED_TARGET attribute
    call takes_linked_pointer(p%next)
    !ERROR: Actual argument with PROTECTED_TARGET pointer 'p' may be associated with pointer dummy argument 'a=' only if it also has the PROTECTED_TARGET attribute
    call takes_linked_pointer_default(p%next)
  end subroutine

  subroutine associate_pointer_targets(p, q, h)
    type(linked_node), pointer, protected_target :: p
    type(linked_node), pointer :: q
    type(linked_holder) :: h
    associate(a => p%next)
      a%value = 1
      call takes_linked_pointer(a)
      call takes_linked_default(a)
      associate(b => a)
        call takes_linked_pointer(b)
        b%value = 2
      end associate
    end associate
    associate(a => q%next)
      call takes_linked_pointer(a)
      a%value = 1
    end associate
    associate(a => h%p%next)
      call takes_linked_pointer(a)
      associate(b => a)
        call takes_linked_pointer(b)
      end associate
    end associate
    associate(a => p)
      call takes_linked_pointer_pt(a)
      call takes_linked_in(a)
      call takes_linked_pointer(a)
      associate(b => a)
        call takes_linked_pointer(b)
      end associate
    end associate
  end subroutine

  subroutine ranked_pointer_descriptors(p, x, y)
    integer, pointer, protected_target :: p(..)
    integer, pointer, protected_target :: r, rs(:)
    integer, target :: x, y(2)
    select rank(a => p)
    ! The associate name has PROTECTED_TARGET only in RANK DEFAULT
    ! (F'2028 11.1.12.3), but definitions through it are still definitions
    ! of the target of p (20.6.7(11)).
    rank(0)
      a => x
      !ERROR: Pointer assignment target 'r' has the PROTECTED_TARGET attribute; pointer 'a' must also have the PROTECTED_TARGET attribute
      a => r
      nullify(a)
      allocate(a, source=1)
      call takes_pointer_pt_out(a)
      !ERROR: ALLOCATE of object associated with PROTECTED_TARGET pointer 'p' requires SOURCE=
      allocate(a)
      !ERROR: Object in DEALLOCATE statement is not deallocatable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
      deallocate(a)
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
      a = 1
      call takes_pointer(a)
    rank(1)
      a => y
      !ERROR: Pointer assignment target 'rs' has the PROTECTED_TARGET attribute; pointer 'a' must also have the PROTECTED_TARGET attribute
      a => rs
      nullify(a)
      allocate(a, source=y)
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
      a(1) = 1
    rank default
      nullify(a)
      call takes_ranked_pointer_pt(a)
      !ERROR: A PROTECTED_TARGET pointer may not be deallocated
      deallocate(a)
    end select
  end subroutine

  subroutine takes_ranked_pointer_pt(a)
    integer, pointer, protected_target, intent(out) :: a(..)
  end subroutine

  subroutine ranked_ordinary_pointer_descriptors(p, x, y)
    integer, pointer :: p(..)
    integer, pointer, protected_target :: r
    integer, target :: x, y(2)
    select rank(a => p)
    rank(0)
      a => x
      nullify(a)
      allocate(a)
      a = 1
      deallocate(a)
      call takes_pointer(a)
      !ERROR: Pointer assignment target 'r' has the PROTECTED_TARGET attribute; pointer 'a' must also have the PROTECTED_TARGET attribute
      a => r
    rank(1)
      a => y
      nullify(a)
      allocate(a, source=y)
      a(1) = 1
      deallocate(a)
    end select
  end subroutine

  subroutine ranked_enclosing_target(p, x)
    type(pt_node), pointer, protected_target :: p(..)
    integer, target :: x
    select rank(a => p)
    rank(0)
      !ERROR: The left-hand side of a pointer assignment is not definable
      !BECAUSE: A pointer subobject of the target of PROTECTED_TARGET pointer 'p' is not definable
      a%child => x
      !ERROR: 'child' may not appear in NULLIFY
      !BECAUSE: A pointer subobject of the target of PROTECTED_TARGET pointer 'p' is not definable
      nullify(a%child)
      !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'a=' is not definable
      !BECAUSE: A pointer subobject of the target of PROTECTED_TARGET pointer 'p' is not definable
      call takes_pointer_pt_out(a%child)
    end select
  end subroutine

  subroutine selected_type_pointer_contexts(p, q, x)
    class(pt_node), pointer, protected_target :: p
    class(pt_node), pointer :: q
    integer, target :: x
    select type(a => q)
    type is(pt_node)
      a%child => x
      nullify(a%child)
      call takes_pointer_pt_out(a%child)
      !ERROR: The left-hand side of a pointer assignment is not definable
      !BECAUSE: 'a' is not a pointer
      a => q
      !ERROR: 'a' may not appear in NULLIFY
      !BECAUSE: 'a' is not a pointer
      nullify(a)
    end select
    select type(a => p)
    type is(pt_node)
      !ERROR: The left-hand side of a pointer assignment is not definable
      !BECAUSE: A pointer subobject of the target of PROTECTED_TARGET pointer 'p' is not definable
      a%child => x
      !ERROR: 'child' may not appear in NULLIFY
      !BECAUSE: A pointer subobject of the target of PROTECTED_TARGET pointer 'p' is not definable
      nullify(a%child)
      !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'a=' is not definable
      !BECAUSE: A pointer subobject of the target of PROTECTED_TARGET pointer 'p' is not definable
      call takes_pointer_pt_out(a%child)
    end select
  end subroutine

  subroutine c872_procedure_component_calls(p, q)
    type(proc_node), pointer, protected_target :: p
    type(proc_node), pointer :: q
    call takes_proc_nonpointer(p%fun)
    call takes_proc_nonpointer(q%fun)
    call takes_proc_pointer_default(q%fun)
    call takes_proc_pointer_in(q%fun)
    call takes_proc_pointer_out(q%fun)
    call takes_proc_pointer_inout(q%fun)
    !ERROR: Actual argument with PROTECTED_TARGET pointer 'p' may not be associated with procedure pointer dummy argument 'proc='
    call takes_proc_pointer_default(p%fun)
    !ERROR: Actual argument with PROTECTED_TARGET pointer 'p' may not be associated with procedure pointer dummy argument 'proc='
    call takes_proc_pointer_in(p%fun)
    !ERROR: Actual argument with PROTECTED_TARGET pointer 'p' may not be associated with procedure pointer dummy argument 'proc='
    call takes_proc_pointer_out(p%fun)
    !ERROR: Actual argument with PROTECTED_TARGET pointer 'p' may not be associated with procedure pointer dummy argument 'proc='
    call takes_proc_pointer_inout(p%fun)
  end subroutine

  subroutine null_mold_protected_pointer_ok()
    integer, pointer, protected_target :: p
    integer, pointer, protected_target :: q
    q => null(mold=p)
  end subroutine

  subroutine owner_ok()
    integer, pointer, protected_target :: p
    p => owned_x
  end subroutine

  function independent_result(box) result(r)
    type(protected_box), intent(in) :: box
    integer, pointer :: r
    r => box%child%value
  end function

  subroutine takes_nonpointer_out(a)
    integer, intent(out) :: a
  end subroutine

  subroutine takes_assumed_rank_out(a)
    integer, intent(out) :: a(..)
  end subroutine

  ! A definition of an associate name is a definition of its selector
  ! (F'2028 20.6.7(11)).
  subroutine alias_definition_contexts(p, h, t, r)
    integer, pointer, protected_target :: p, r(..)
    type(pt_holder) :: h
    class(pt_node), pointer, protected_target :: t
    integer :: j
    associate (a => p)
      !ERROR: 'a' may not be used as a DO variable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
      do a = 1, 2
      end do
      !ERROR: The target of PROTECTED_TARGET pointer 'p' is not definable
      print *, (j, a = 1, 2)
      !ERROR: Input variable 'a' is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
      read *, a
      !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'a=' is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
      call takes_nonpointer_out(a)
      associate (b => a)
        !ERROR: 'b' may not be used as a DO variable
        !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
        do b = 1, 2
        end do
      end associate
    end associate
    associate (a => h%p%value)
      !ERROR: 'a' may not be used as a DO variable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
      do a = 1, 2
      end do
      !ERROR: Input variable 'a' is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
      read *, a
    end associate
    associate (a => protected_function_result())
      !ERROR: 'a' may not be used as a DO variable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
      do a = 1, 2
      end do
      !ERROR: Input variable 'a' is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
      read *, a
    end associate
    select type (a => t)
    type is (pt_node)
      !ERROR: Input variable 'a' is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 't' is not definable
      read *, a%value
      !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'a=' is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 't' is not definable
      call takes_nonpointer_out(a%value)
    end select
    select rank (a => r)
    rank (0)
      !ERROR: 'a' may not be used as a DO variable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'r' is not definable
      do a = 1, 2
      end do
      !ERROR: Input variable 'a' is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'r' is not definable
      read *, a
      !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'a=' is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'r' is not definable
      call takes_nonpointer_out(a)
    rank default
      !ERROR: Actual argument with PROTECTED_TARGET pointer 'a' may be associated with non-pointer dummy argument 'a=' only if it has INTENT(IN)
      !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'a=' is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'r' is not definable
      call takes_assumed_rank_out(a)
    end select
  end subroutine

  ! Other appearances of an associate name are not restricted, since it
  ! lacks PROTECTED_TARGET outside RANK DEFAULT (F'2028 11.1.3.3, 11.1.12.3):
  ! the ASSOCIATE and SELECT TYPE names are not pointers, and the RANK (0)
  ! name is a pointer without PROTECTED_TARGET.  A pointer component that it
  ! selects keeps its own PROTECTED_TARGET attribute.
  subroutine alias_nondefinition_ok(p, h, t, r)
    integer, pointer, protected_target :: p, r(..)
    type(pt_holder), target :: h
    class(pt_node), pointer, protected_target :: t
    integer, pointer :: q
    type(pt_node), pointer :: n
    integer :: j
    external :: implicit_sub
    associate (a => p)
      q => a
      call takes_pointer(a)
      call takes_nonpointer_default(a)
      call implicit_sub(a)
      j = a
      do j = a, 2
      end do
    end associate
    select type (a => t)
    type is (pt_node)
      q => a%value
      call takes_nonpointer_default(a%value)
    end select
    select rank (a => r)
    rank (0)
      q => a
      call takes_nonpointer_default(a)
    end select
    associate (a => h)
      n => a%q
      !ERROR: Pointer assignment target 'p' has the PROTECTED_TARGET attribute; pointer 'n' must also have the PROTECTED_TARGET attribute
      n => a%p
      !ERROR: Actual argument with PROTECTED_TARGET pointer 'child' may be associated with pointer dummy argument 'a=' only if it also has the PROTECTED_TARGET attribute
      call takes_pointer(a%p%child)
    end associate
  end subroutine
end module

subroutine c864_conflict_case()
  use m15_protected_target, only: owned_x
  integer, pointer, protected_target :: p
  !ERROR: Use-associated nonpointer PROTECTED target 'owned_x' may not be pointer-assigned through PROTECTED_TARGET pointer 'p'
  p => owned_x
end subroutine

subroutine c864_component_targets()
  use m15_protected_target, only: owned_box, linked_node, independent_result
  integer, pointer, protected_target :: r, rs(:)
  integer, pointer :: ordinary(:)
  type(linked_node), pointer, protected_target :: p
  p => owned_box%child
  p => owned_box%child%next
  r => owned_box%child%value
  r => independent_result(owned_box)
  ordinary => owned_box%values
  !ERROR: Use-associated nonpointer PROTECTED target 'owned_box' may not be pointer-assigned through PROTECTED_TARGET pointer 'r'
  r => owned_box%fixed%value
  !ERROR: Use-associated nonpointer PROTECTED target 'owned_box' may not be pointer-assigned through PROTECTED_TARGET pointer 'rs'
  rs => owned_box%values(2:)
  associate(a => owned_box%child)
    p => a
    associate(b => a)
      p => b
      r => b%value
    end associate
  end associate
  associate(a => owned_box%fixed)
    !ERROR: Use-associated nonpointer PROTECTED target 'owned_box' may not be pointer-assigned through PROTECTED_TARGET pointer 'r'
    r => a%value
  end associate
  associate(a => owned_box%values(2:))
    !ERROR: Use-associated nonpointer PROTECTED target 'owned_box' may not be pointer-assigned through PROTECTED_TARGET pointer 'rs'
    rs => a
  end associate
end subroutine

! PROTECTED_TARGET is not a characteristic of a dummy data object or of a
! function result (F'2028 15.3.2.2, 15.3.3), but an explicit interface still
! applies its constraints to calls.
module m16_protected_target_characteristics
  implicit none
  type :: base
  contains
    procedure, nopass :: take => base_take
    procedure, nopass :: get => base_get
  end type
  type, extends(base) :: ext
  contains
    procedure, nopass :: take => ext_take
    procedure, nopass :: get => ext_get
  end type
  interface
    module subroutine separate_take(a)
      integer, pointer, protected_target, intent(in) :: a
    end subroutine
    module function separate_get() result(r)
      integer, pointer :: r
    end function
  end interface
  abstract interface
    subroutine take_pt(a)
      integer, pointer, protected_target, intent(in) :: a
    end subroutine
    subroutine take_plain(a)
      integer, pointer, intent(in) :: a
    end subroutine
    function get_pt() result(r)
      integer, pointer, protected_target :: r
    end function
    function get_plain() result(r)
      integer, pointer :: r
    end function
  end interface
  !ERROR: Generic 'take' may not have specific procedures 'base_take' and 'ext_take' as their interfaces are not distinguishable
  interface take
    procedure base_take, ext_take
  end interface
contains
  subroutine base_take(a)
    integer, pointer, protected_target, intent(in) :: a
  end subroutine
  subroutine ext_take(a)
    integer, pointer, intent(in) :: a
  end subroutine
  function base_get() result(r)
    integer, pointer :: r
    r => null()
  end function
  function ext_get() result(r)
    integer, pointer, protected_target :: r
    r => null()
  end function
  subroutine takes_take_pt(f)
    procedure(take_pt) :: f
  end subroutine
  subroutine takes_take_plain(f)
    procedure(take_plain) :: f
  end subroutine
  subroutine takes_get_pt(f)
    procedure(get_pt) :: f
  end subroutine
  subroutine takes_get_plain(f)
    procedure(get_plain) :: f
  end subroutine
  subroutine procedure_matching_ok()
    procedure(take_pt), pointer :: tp
    procedure(take_plain), pointer :: tq
    procedure(get_pt), pointer :: gp
    procedure(get_plain), pointer :: gq
    tp => ext_take
    tq => base_take
    gp => base_get
    gq => ext_get
    call takes_take_pt(ext_take)
    call takes_take_plain(base_take)
    call takes_get_pt(base_get)
    call takes_get_plain(ext_get)
  end subroutine
  subroutine calls_through_interfaces(p, tp, tq, gp)
    integer, pointer, protected_target :: p
    procedure(take_pt) :: tp
    procedure(take_plain) :: tq
    procedure(get_pt) :: gp
    integer, pointer :: q
    type(ext) :: e
    call tp(p)
    call base_take(p)
    !ERROR: Actual argument with PROTECTED_TARGET pointer 'p' may be associated with pointer dummy argument 'a=' only if it also has the PROTECTED_TARGET attribute
    call tq(p)
    !ERROR: Actual argument with PROTECTED_TARGET pointer 'p' may be associated with pointer dummy argument 'a=' only if it also has the PROTECTED_TARGET attribute
    call e%take(p)
    !ERROR: Pointer assignment target is function 'gp' result with the PROTECTED_TARGET attribute; pointer 'q' must also have the PROTECTED_TARGET attribute
    q => gp()
    !ERROR: Pointer assignment target is function 'ext_get' result with the PROTECTED_TARGET attribute; pointer 'q' must also have the PROTECTED_TARGET attribute
    q => e%get()
  end subroutine
end module

submodule(m16_protected_target_characteristics) m16_protected_target_body
contains
  module subroutine separate_take(a)
    integer, pointer, intent(in) :: a
  end subroutine
  module function separate_get() result(r)
    integer, pointer, protected_target :: r
    r => null()
  end function
end submodule

subroutine m16_external(a)
  integer, pointer, intent(in) :: a
end subroutine

subroutine m16_external_caller(p)
  integer, pointer, protected_target :: p
  interface
    subroutine m16_external(a)
      integer, pointer, protected_target, intent(in) :: a
    end subroutine
  end interface
  call m16_external(p)
end subroutine

! C867 applies to a PROTECTED_TARGET pointer that is itself in a common
! block; a sequence type with such a pointer component may be a common
! block object (F'2028 8.11.3.1, C8143, C8144). PROTECTED_TARGET is a
! component attribute, so it distinguishes sequence types (7.5.2.5).
subroutine m17_common_holder()
  type :: seq_holder
    sequence
    integer, pointer, protected_target :: p
  end type
  type(seq_holder) :: h
  common /m17_blk/ h
end subroutine

subroutine m17_caller()
  type :: seq_holder
    sequence
    integer, pointer, protected_target :: p
  end type
  type(seq_holder) :: h
  interface
    subroutine m17_take_pt(h)
      type :: seq_holder
        sequence
        integer, pointer, protected_target :: p
      end type
      type(seq_holder) :: h
    end subroutine
    subroutine m17_take_plain(h)
      type :: seq_holder
        sequence
        integer, pointer :: p
      end type
      type(seq_holder) :: h
    end subroutine
  end interface
  call m17_take_pt(h)
  !ERROR: Actual argument type 'seq_holder' is not compatible with dummy argument type 'seq_holder'
  call m17_take_plain(h)
end subroutine

! PROTECTED_TARGET is not a characteristic (15.3.2.2), so it does not take
! part in generic resolution (15.5.5.2). C872 and C874 apply to the specific
! procedure that is selected; an intrinsic procedure, an elemental specific
! or intrinsic assignment is not chosen instead.
module m18_protected_target_generics
  implicit none
  type :: t
    integer :: v
  contains
    procedure :: bump
    procedure :: peek
    generic :: gen_bump => bump
    generic :: gen_peek => peek
  end type
  abstract interface
    subroutine iface()
    end subroutine
  end interface
  type :: proc_holder
    procedure(iface), pointer, nopass :: fun
  end type
  interface abs
    module procedure abs_inout
  end interface
  interface g_rank
    module procedure g_arr_inout, g_elem_in
  end interface
  interface g_ptr
    module procedure g_ptr_int, g_ptr_real_pt
  end interface
  interface g_proc
    module procedure g_proc_ptr, g_proc_int
  end interface
  interface operator(.plus.)
    !WARNING: In .PLUS. function 'plus_default', dummy argument 'a' should have INTENT(IN) or VALUE attribute [-Wdefined-operator-args]
    !WARNING: In .PLUS. function 'plus_default', dummy argument 'b' should have INTENT(IN) or VALUE attribute [-Wdefined-operator-args]
    module procedure plus_default, plus_in
  end interface
  interface assignment(=)
    !WARNING: In defined assignment subroutine 'assign_t', second dummy argument 'b' should have INTENT(IN) or VALUE attribute [-Wdefined-operator-args]
    module procedure assign_int, assign_t
  end interface
contains
  subroutine bump(this)
    class(t), intent(inout) :: this
  end subroutine
  subroutine peek(this)
    class(t), intent(in) :: this
  end subroutine
  integer function abs_inout(n)
    integer, intent(inout) :: n
    abs_inout = n
  end function
  subroutine g_arr_inout(arr)
    integer, intent(inout) :: arr(:)
  end subroutine
  elemental subroutine g_elem_in(a)
    integer, intent(in) :: a
  end subroutine
  subroutine g_ptr_int(ip)
    integer, pointer, intent(in) :: ip
  end subroutine
  subroutine g_ptr_real_pt(rp)
    real, pointer, protected_target, intent(in) :: rp
  end subroutine
  subroutine g_proc_ptr(pp)
    procedure(iface), pointer, intent(in) :: pp
  end subroutine
  subroutine g_proc_int(j)
    integer, intent(in) :: j
  end subroutine
  integer function plus_default(a, b)
    type(t) :: a, b
    plus_default = 0
  end function
  integer function plus_in(a, j)
    type(t), intent(in) :: a
    integer, intent(in) :: j
    plus_in = j
  end function
  subroutine assign_int(a, j)
    type(t), intent(out) :: a
    integer, intent(in) :: j
  end subroutine
  subroutine assign_t(a, b)
    type(t), intent(out) :: a
    type(t) :: b
  end subroutine
  subroutine generic_calls(np, ap, rp, tp, hp, q, x)
    integer, pointer, protected_target :: np, ap(:)
    real, pointer, protected_target :: rp
    type(t), pointer, protected_target :: tp
    type(proc_holder), pointer, protected_target :: hp
    integer, pointer :: q
    type(t), intent(out) :: x
    integer :: n
    !ERROR: Actual argument with PROTECTED_TARGET pointer 'np' may be associated with non-pointer dummy argument 'n=' only if it has INTENT(IN)
    !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'n=' is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'np' is not definable
    n = abs(np)
    !ERROR: Actual argument with PROTECTED_TARGET pointer 'ap' may be associated with non-pointer dummy argument 'arr=' only if it has INTENT(IN)
    !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'arr=' is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'ap' is not definable
    call g_rank(ap)
    !ERROR: Actual argument with PROTECTED_TARGET pointer 'np' may be associated with pointer dummy argument 'ip=' only if it also has the PROTECTED_TARGET attribute
    call g_ptr(np)
    call g_ptr(rp)
    call g_ptr(q)
    !ERROR: Actual argument with PROTECTED_TARGET pointer 'hp' may not be associated with procedure pointer dummy argument 'pp='
    call g_proc(hp%fun)
    !ERROR: Actual argument with PROTECTED_TARGET pointer 'tp' may be associated with non-pointer dummy argument 'a=' only if it has INTENT(IN)
    !ERROR: Actual argument with PROTECTED_TARGET pointer 'tp' may be associated with non-pointer dummy argument 'b=' only if it has INTENT(IN)
    n = tp .plus. tp
    n = tp .plus. 1
    !ERROR: Actual argument with PROTECTED_TARGET pointer 'tp' may be associated with non-pointer dummy argument 'a=' only if it has INTENT(IN)
    !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'a=' is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'tp' is not definable
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'tp' is not definable
    tp = 1
    ! The right-hand side is passed as (tp), not as the pointer (15.4.3.4.3)
    x = tp
    !ERROR: Actual argument with PROTECTED_TARGET pointer 'tp' may be associated with non-pointer dummy argument 'this=' only if it has INTENT(IN)
    !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'this=' is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'tp' is not definable
    call tp%gen_bump()
    call tp%gen_peek()
  end subroutine
end module

! C872 has no exception for intrinsic procedures (15.5.2.4, 20.6.8): FPTR of
! C_F_POINTER and FSTRPTR of C_F_STRPOINTER are INTENT(OUT) pointer dummy
! arguments without PROTECTED_TARGET (19.2.3.3, 19.2.3.5).
subroutine c872_intrinsic_pointer_dummies(c, chars)
  use iso_c_binding
  implicit none
  type(c_ptr), intent(in) :: c
  character(kind=c_char), intent(in), target :: chars(10)
  integer, pointer, protected_target :: p, a(:)
  character(:), pointer, protected_target :: sp
  integer, pointer :: q
  character(:), pointer :: sq
  integer :: n
  !ERROR: Actual argument with PROTECTED_TARGET pointer 'p' may be associated with pointer dummy argument 'fptr=' only if it also has the PROTECTED_TARGET attribute
  call c_f_pointer(c, p)
  !ERROR: Actual argument with PROTECTED_TARGET pointer 'sp' may be associated with pointer dummy argument 'fstrptr=' only if it also has the PROTECTED_TARGET attribute
  call c_f_strpointer(chars, sp)
  call c_f_pointer(c, q)
  call c_f_strpointer(chars, sq)
  if (associated(p) .and. associated(p, q)) then
    n = size(a) + lbound(a, 1) + ubound(a, 1) + len(sp)
  end if
  p => null(mold=p)
end subroutine

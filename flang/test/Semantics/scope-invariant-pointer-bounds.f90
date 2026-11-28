! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s

! INTENT(IN) prevents pointers from being reassociated but doesn't prevent
! mutation of the pointer's target.

module bounds
  type bounds_t
    integer, pointer :: n
  end type
  type inner_t
    integer :: n
  end type
  type outer_t
    type(inner_t), pointer :: p
  end type
contains
  ! CHECK-LABEL: SUBROUTINE pointer_bound
  ! Don't fold SIZE at compile time. n could be mutated.
  ! CHECK: PRINT *, size(a,dim=1,kind=8)
  subroutine pointer_bound(n, a)
    integer, pointer, intent(in) :: n
    integer :: a(n:n+42)
    call mutate(n)
    print *, size(a, kind=8)
  end subroutine

  ! CHECK-LABEL: SUBROUTINE component_bound
  ! Don't fold SIZE at compile time. b%n could be mutated.
  ! CHECK: PRINT *, size(a,dim=1,kind=8)
  subroutine component_bound(b, a)
    type(bounds_t), intent(in) :: b
    integer :: a(b%n:b%n+42)
    call mutate(b%n)
    print *, size(a, kind=8)
  end subroutine

  ! CHECK-LABEL: SUBROUTINE invariant_bound
  ! Safe to fold SIZE at compile time
  ! CHECK: PRINT *, max(0_8,{{.*}}n{{.*}})
  subroutine invariant_bound(n, a)
    integer, intent(in) :: n
    integer :: a(n:n+42)
    ! mutation of n not allowed
    print *, size(a, kind=8)
  end subroutine

  ! CHECK-LABEL: SUBROUTINE nointent_bound
  ! No INTENT(IN). Unsafe to fold SIZE.
  ! CHECK: PRINT *, size(a,dim=1,kind=8)
  subroutine nointent_bound(n, a)
    integer :: n
    integer :: a(n:n+42)
    call mutate(n)
    print *, size(a, kind=8)
  end subroutine

  ! CHECK-LABEL: SUBROUTINE len_bound
  ! Don't fold LEN at compile time. n could be mutated.
  ! CHECK: PRINT *, len(s)
  subroutine len_bound(n, s)
    integer, pointer, intent(in) :: n
    character(len=n) :: s
    print *, len(s)
  end subroutine

  ! CHECK-LABEL: SUBROUTINE nested_component_bound
  ! Don't fold SIZE at compile time. b%p%n could be mutated.
  ! CHECK: PRINT *, size(a,dim=1,kind=8)
  subroutine nested_component_bound(b, a)
    type(outer_t), intent(in) :: b
    real :: a(b%p%n)
    print *, size(a, kind=8)
  end subroutine
end module

! Descriptor inquiries depend on association, not on target values.

module descriptor_bounds
  type inner_t
    integer, pointer :: q(:)
  end type
  type outer_t
    type(inner_t), pointer :: p
  end type
  integer, pointer :: host_pointer(:)
contains
  ! CHECK-LABEL: SUBROUTINE nested_size
  ! Don't fold SIZE at compile time. b%p%q could be reassociated.
  ! CHECK: PRINT *, size(a,dim=1,kind=8)
  subroutine nested_size(b, a)
    type(outer_t), intent(in) :: b
    real :: a(size(b%p%q))
    print *, size(a, kind=8)
  end subroutine

  ! CHECK-LABEL: SUBROUTINE nested_bounds
  ! Don't fold SIZE at compile time. b%p%q's bounds could change.
  ! CHECK: PRINT *, size(a,dim=1,kind=8), size(c,dim=1,kind=8)
  subroutine nested_bounds(b, a, c)
    type(outer_t), intent(in) :: b
    real :: a(lbound(b%p%q, 1)), c(ubound(b%p%q, 1))
    print *, size(a, kind=8), size(c, kind=8)
  end subroutine

  ! CHECK-LABEL: SUBROUTINE pointer_base
  ! Don't fold SIZE at compile time. b%q could be reassociated.
  ! CHECK: PRINT *, size(a,dim=1,kind=8)
  subroutine pointer_base(b, a)
    type(inner_t), pointer, intent(in) :: b
    real :: a(size(b%q))
    print *, size(a, kind=8)
  end subroutine

  ! CHECK-LABEL: SUBROUTINE mutable_subscript
  ! Don't fold SIZE at compile time. i could be mutated.
  ! CHECK: PRINT *, size(a,dim=1,kind=8)
  subroutine mutable_subscript(b, i, a)
    type(inner_t), intent(in) :: b(:)
    integer :: i
    real :: a(size(b(i)%q))
    print *, size(a, kind=8)
  end subroutine

  ! CHECK-LABEL: SUBROUTINE host_pointer_bound
  ! Don't fold SIZE at compile time. host_pointer could be reassociated.
  ! CHECK: PRINT *, size(a,dim=1,kind=8)
  subroutine host_pointer_bound(a)
    real :: a(size(host_pointer))
    print *, size(a, kind=8)
  end subroutine

  ! CHECK-LABEL: SUBROUTINE invariant_pointer
  ! Safe to fold SIZE at compile time. p cannot be reassociated.
  ! CHECK: PRINT *, max(0_8,size(p,dim=1,kind=8))
  subroutine invariant_pointer(p, a)
    integer, pointer, intent(in) :: p(:)
    real :: a(size(p))
    print *, size(a, kind=8)
  end subroutine

  ! CHECK-LABEL: SUBROUTINE invariant_component
  ! Safe to fold SIZE at compile time. b%q cannot be reassociated.
  ! CHECK: PRINT *, max(0_8,size(b%q,dim=1,kind=8))
  subroutine invariant_component(b, a)
    type(inner_t), intent(in) :: b
    real :: a(size(b%q))
    print *, size(a, kind=8)
  end subroutine

  ! CHECK-LABEL: SUBROUTINE invariant_subscript
  ! Safe to fold SIZE at compile time. Neither i nor b(i)%q can change.
  ! CHECK: PRINT *, max(0_8,size(b({{.*}}i{{.*}})%q,dim=1,kind=8))
  subroutine invariant_subscript(b, i, a)
    type(inner_t), intent(in) :: b(:)
    integer, intent(in) :: i
    real :: a(size(b(i)%q))
    print *, size(a, kind=8)
  end subroutine
end module

! A fixed shape does not make SIZE invariant when DIM can change.
module size_dim
contains
  ! CHECK-LABEL: SUBROUTINE pointer_dim
  ! CHECK: PRINT *, size(a,dim=1,kind=8)
  subroutine pointer_dim(d, m, a)
    integer, pointer, intent(in) :: d
    real, intent(in) :: m(2, 4)
    real :: a(size(m, dim=d))
    d = 2
    print *, size(a, kind=8)
  end subroutine

  ! CHECK-LABEL: SUBROUTINE mutable_dim
  ! CHECK: PRINT *, size(a,dim=1,kind=8)
  subroutine mutable_dim(d, m, a)
    integer :: d
    real, intent(in) :: m(2, 4)
    real :: a(size(m, dim=d))
    d = 2
    print *, size(a, kind=8)
  end subroutine

  ! CHECK-LABEL: SUBROUTINE invariant_dim
  ! CHECK: PRINT *, max(0_8,__builtin_int(size(m,dim=d),kind=8))
  subroutine invariant_dim(d, m, a)
    integer, intent(in) :: d
    real, intent(in) :: m(2, 4)
    real :: a(size(m, dim=d))
    print *, size(a, kind=8)
  end subroutine

  ! CHECK-LABEL: SUBROUTINE absent_dim
  ! CHECK: PRINT *, 8_8
  subroutine absent_dim(m, a)
    real, intent(in) :: m(2, 4)
    real :: a(size(m))
    print *, size(a, kind=8)
  end subroutine
end module

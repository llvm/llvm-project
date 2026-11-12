! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s

! INTENT(IN) prevents pointers from being reassociated but doesn't prevent
! mutation of the pointer's target.

module bounds
  type bounds_t
    integer, pointer :: n
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
end module

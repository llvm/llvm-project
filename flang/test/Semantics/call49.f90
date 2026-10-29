! RUN: %python %S/test_errors.py %s %flang_fc1 -Wintent-in-actual-for-default-intent -Werror

! An INTENT(IN) dummy argument must not be defined during the execution of its
! procedure (F'2023 8.5.10 p2); passing one to a dummy argument that has no
! INTENT attribute risks exactly that.

module m
  type t
    integer :: n
  end type
 contains
  subroutine noIntent(n)
    integer :: n
  end subroutine
  subroutine intentIn(n)
    integer, intent(in) :: n
  end subroutine
  subroutine byValue(n)
    integer, value :: n
  end subroutine
  pure subroutine pureSub(n)
    integer, intent(in) :: n
  end subroutine
  subroutine test(k, j, v, arr, dt)
    integer, intent(in) :: k, arr(:)
    integer, intent(in out) :: j
    integer, intent(in), value :: v
    type(t), intent(in) :: dt
    !WARNING: INTENT(IN) dummy argument 'k' is associated with dummy argument 'n=', which has no INTENT attribute and could be defined [-Wintent-in-actual-for-default-intent]
    call noIntent(k)
    !WARNING: INTENT(IN) dummy argument 'arr' is associated with dummy argument 'n=', which has no INTENT attribute and could be defined [-Wintent-in-actual-for-default-intent]
    call noIntent(arr(1))
    !WARNING: INTENT(IN) dummy argument 'dt' is associated with dummy argument 'n=', which has no INTENT attribute and could be defined [-Wintent-in-actual-for-default-intent]
    call noIntent(dt%n)
    call intentIn(k) ! ok
    call byValue(k) ! ok, cannot be defined
    call pureSub(k) ! ok
    call noIntent(j) ! ok, INTENT(IN OUT)
    call noIntent(v) ! ok, VALUE actual is a local copy
    call noIntent(k + 0) ! ok, not a variable
   contains
    subroutine inner
      !WARNING: INTENT(IN) dummy argument 'k' is associated with dummy argument 'n=', which has no INTENT attribute and could be defined [-Wintent-in-actual-for-default-intent]
      call noIntent(k) ! host association
    end subroutine
  end subroutine
end module

! The motivating case: the callee's interface is implicit at the point of
! the call, but its definition is available in the same source file.
subroutine callsExternal(k)
  integer, intent(in) :: k
  !WARNING: INTENT(IN) dummy argument 'k' is associated with dummy argument 'n=', which has no INTENT attribute and could be defined [-Wintent-in-actual-for-default-intent]
  call externalNoIntent(k)
end subroutine

subroutine externalNoIntent(n)
  integer :: n
  n = 3
end subroutine

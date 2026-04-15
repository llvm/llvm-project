! RUN: %flang_fc1 -fsafe-trampoline -emit-llvm -o - %s | FileCheck %s
!
! Test that -fsafe-trampoline generates calls to the runtime
! trampoline pool instead of stack-based trampolines.
!
! Test cases cover trampolines at the top-level block, used inside an IF block,
! and used inside a DO loop.

! CHECK-LABEL: define {{.*}}@host_
! CHECK: call {{.*}}@_FortranATrampolineInit
! CHECK: call {{.*}}@_FortranATrampolineAdjust
! CHECK: call {{.*}}@_FortranATrampolineFree

! CHECK-LABEL: define {{.*}}@host_in_if_
! CHECK: call {{.*}}@_FortranATrampolineInit
! CHECK: call {{.*}}@_FortranATrampolineAdjust
! CHECK: call {{.*}}@_FortranATrampolineFree

! CHECK-LABEL: define {{.*}}@host_in_do_loop_
! CHECK: call {{.*}}@_FortranATrampolineInit
! CHECK: call {{.*}}@_FortranATrampolineAdjust
! CHECK: call {{.*}}@_FortranATrampolineFree

! CHECK-LABEL: define {{.*}}@host_branch_
! CHECK: call {{.*}}@_FortranATrampolineInit
! CHECK: call {{.*}}@_FortranATrampolineAdjust
! CHECK: call {{.*}}@_FortranATrampolineFree

module other
  abstract interface
     function callback()
       integer :: callback
     end function callback
  end interface
  contains
  subroutine foo(fptr)
    procedure(callback), pointer :: fptr
    print *, fptr()
  end subroutine foo
  subroutine bar(fproc)
    procedure(callback) :: fproc
    print *, fproc()
  end subroutine bar
end module other

! Test 1: trampoline at top-level block (baseline).
subroutine host(local)
  use other
  integer :: local
  procedure(callback), pointer :: fptr
  fptr => callee
  call foo(fptr)
  return

  contains

  function callee()
    integer :: callee
    callee = local
  end function callee
end subroutine host

! Test 2: trampoline used inside an IF block.
subroutine host_in_if(local, flag)
  use other
  integer :: local
  logical :: flag
  procedure(callback), pointer :: fptr
  fptr => callee
  if (flag) then
    call foo(fptr)
  end if
  return

  contains

  function callee()
    integer :: callee
    callee = local
  end function callee
end subroutine host_in_if

! Test 3: trampoline used inside a DO loop.
subroutine host_in_do_loop(local, n)
  use other
  integer :: local
  integer :: n
  integer :: i
  procedure(callback), pointer :: fptr
  fptr => callee
  do i = 1, n
    call foo(fptr)
  end do
  return

  contains

  function callee()
    integer :: callee
    callee = local + i
  end function callee
end subroutine host_in_do_loop

! Test 4: emboxproc generated inside a branch (internal procedure passed
! directly as actual argument inside an IF block).
subroutine host_branch(local, flag)
  use other
  integer :: local
  logical :: flag
  if (flag) call bar(callee)
  return

  contains

  function callee()
    integer :: callee
    callee = local
  end function callee
end subroutine host_branch

program main
  call host(10)
  call host_in_if(20, .true.)
  call host_in_do_loop(30, 3)
  call host_branch(40, .true.)
end program main

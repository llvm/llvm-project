! UNSUPPORTED: system-windows
! UNSUPPORTED: offload-cuda
! UNSUPPORTED: system-darwin

! Verify that -fsafe-trampoline produces an executable whose
! GNU_STACK program header is RW (not RWE), proving W^X compliance.
! The legacy stack-trampoline path requires an executable stack; the
! runtime trampoline pool does not.

! RUN: %flang %isysroot -fsafe-trampoline -L"%libdir" %s -o %t
! RUN: llvm-readelf -lW %t | FileCheck %s

! Ensure GNU_STACK exists and has RW flags (no E).
! CHECK: GNU_STACK
! CHECK-SAME: RW
! CHECK-NOT: RWE

subroutine host_proc(x, res)
  implicit none
  integer, intent(in) :: x
  integer, intent(out) :: res

  interface
    function f_iface() result(r)
      integer :: r
    end function
  end interface

  procedure(f_iface), pointer :: fptr
  fptr => inner
  res = fptr()

contains
  function inner() result(r)
    integer :: r
    r = x + 1
  end function
end subroutine

program test_gnustack
  implicit none
  integer :: result
  call host_proc(1, result)
  print *, result
end program

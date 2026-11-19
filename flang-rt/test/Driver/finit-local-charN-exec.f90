! Executable regression test for -finit-local= with runtime-length CHARACTER.
! Verifies that every byte of a character(n) local is initialized to the
! requested pattern, and that the empty-string (n=0) case runs without error.
!
! UNSUPPORTED: offload-cuda
!
! RUN: %flang %isysroot -L"%libdir" -finit-local=0xAA %s -o %t
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t

program test_finit_local_charN
  implicit none

  call check_charN(5)   ! n > 0: all bytes must be 0xAA
  call check_charN(1)   ! single byte
  call check_charN(0)   ! empty string: no bytes to check, must not crash
end program

subroutine check_charN(n)
  integer, intent(in) :: n
  character(n) :: x
  integer :: i
  ! Inspect each byte through an equivalenced integer array.
  ! For n == 0 the loop body is never entered.
  do i = 1, n
    if (ichar(x(i:i)) /= int(z'AA')) then
      write(*,*) 'FAIL: byte', i, 'of character(', n, ') =', ichar(x(i:i))
      stop 1
    end if
  end do
end subroutine

! Executable regression test for -finit-local= with kind-2 CHARACTER.
! Verifies that each code unit of a character(kind=2) local equals
! int(z'AAAA'): the 0xAA byte pattern covers both bytes of every
! 2-byte code unit so the full 16-bit value must be 0xAAAA (= 43690).
!
! UNSUPPORTED: offload-cuda
!
! RUN: %flang %isysroot -L"%libdir" -finit-local=0xAA %s -o %t
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t

program test_finit_local_char2
  implicit none

  call check_char2_fixed()
  call check_char2_runtime(3)
  call check_char2_runtime(1)
  call check_char2_runtime(0)   ! empty: must not crash
end program

! Fixed-length character(kind=2, len=3): 3 code units, each must be 0xAAAA.
subroutine check_char2_fixed()
  character(kind=2, len=3) :: x
  integer :: i
  do i = 1, 3
    if (ichar(x(i:i)) /= int(z'AAAA')) then
      write(*,*) 'FAIL check_char2_fixed: code unit', i, '=', ichar(x(i:i)), &
                 'expected', int(z'AAAA')
      error stop 1
    end if
  end do
end subroutine

! Runtime-length character(kind=2, len=n): each code unit must be 0xAAAA.
subroutine check_char2_runtime(n)
  integer, intent(in) :: n
  character(kind=2, len=n) :: x
  integer :: i
  do i = 1, n
    if (ichar(x(i:i)) /= int(z'AAAA')) then
      write(*,*) 'FAIL check_char2_runtime: code unit', i, &
                 'of character(kind=2,len=', n, ') =', ichar(x(i:i)), &
                 'expected', int(z'AAAA')
      error stop 1
    end if
  end do
end subroutine

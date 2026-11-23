! UNSUPPORTED: offload-cuda
! UNSUPPORTED: system-windows

! Addresses bug: [flang-rt] OPEN after the runtime has closed all units segfaults when FORT_CONVERT_UNIT lists 
! https://github.com/llvm/llvm-project/issues/243040
!
!
! RUN: %flang %isysroot -L"%libdir" %s -o %t
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" FORT_CONVERT_UNIT="swap:11" %t | FileCheck %s
!
! CHECK: PASS

module issue_243040_mod
  use iso_c_binding
contains
  subroutine late() bind(c)
    open (11, file='late.dat', form='unformatted', status='replace')
    write (11) 42
    close (11)
    print*, 'OPEN/WRITE/CLOSE successful.'
    print *, 'PASS'
    stop
  end subroutine
end module issue_243040_mod

program issue_243040
  use issue_243040_mod
  interface
    integer(c_int) function atexit(f) bind(c, name='atexit')
      import :: c_int, c_funptr
      type(c_funptr), value :: f
    end function
  end interface
  integer :: rc
  rc = atexit(c_funloc(late)) ! registered before the first Fortran I/O
  print *, 'main done'
end program issue_243040

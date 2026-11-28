! UNSUPPORTED: offload-cuda
!
! Verify that the reading of a record beyond end of file in a direct access
! file is flagged as an error.
!
! Reported as issue 193434 (https://github.com/llvm/llvm-project/issues/193434).
!
! RUN: %flang %isysroot -L"%libdir" %s -o %t
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t | FileCheck %s
!
! CHECK: PASS

program direct_read_endfile
  use iso_fortran_env
  implicit none
  real, dimension(16) :: in, out
  integer :: iostat
  character(len=80) :: iomsg

  open(10, recl=size(out)*4, access="direct", iostat=iostat, &
       form="unformatted", status="unknown", err=100, iomsg=iomsg)
  if (iostat /= 0) then
    print*, 'Should not have fallen through with iostat /= 0 after open'
    print*, 'iostat=', iostat
    print*, 'iomsg=', iomsg
    print*, 'FAIL'
    stop 1
  end if

  write(10, rec=1, iostat=iostat, err=200, iomsg=iomsg) out
  if (iostat /= 0) then
    print*, 'Should not have fallen through with iostat /= 0 after write'
    print*, 'iostat=', iostat
    print*, 'iomsg=', iomsg
    print*, 'FAIL'
    stop 2
  end if

  read(10, rec=1, iostat=iostat, err=300, iomsg=iomsg)
  if (iostat /= 0) then
    print*, 'Should not have fallen through with iostat /= 0 after read rec=1'
    print*, 'iostat=', iostat
    print*, 'iomsg=', iomsg
    print*, 'FAIL'
    stop 3
  end if

  read(10, rec=2, iostat=iostat, err=400, iomsg=iomsg)
  print*, 'Should not have fallen through - this is original bug of issue 193434'
  print*, 'iostat=', iostat
  print*, 'iomsg=', iomsg
  print*, 'FAIL'
  stop 4

! Error states
! 100 - open failed
100 continue
  print*, 'Open should not have failed'
  print*, 'iostat=', iostat
  print*, 'iomsg=', iomsg

!200 - write of rec=1 failed
200 continue
  print*, 'write rec=1 should not have failed'
  print*, 'iostat=', iostat
  print*, 'iomsg=', iomsg
  stop 5

! 300 - read of rec=1 failed
300 continue
  print*, 'read rec=1 should not have failed'
  print*, 'iostat=', iostat
  print*, 'iomsg=', iomsg
  stop 6

! 400 - execution path should have taken us here by reading an undefined record.
400 continue
  if (iostat == 0) then
    print*, 'rec=2 not found, but iostat == 0'
    print*, 'iostat=', iostat
    print*, 'iomsg=', iomsg
    stop 5
  end if

  print*, 'PASS'
end program direct_read_endfile

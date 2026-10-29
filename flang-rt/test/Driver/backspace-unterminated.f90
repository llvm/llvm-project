! The last record of a formatted sequential file need not be terminated
! by a newline; BACKSPACE must be able to reposition over such a record.

! RUN: %flang %isysroot -L"%libdir" %s -o %t
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t | FileCheck %s

! CHECK: PASS: unterm1
! CHECK: PASS: unterm2

program backspace_unterminated
  implicit none
  call check('unterm1', 'ABCDEFGHIJ', 1)
  call check('unterm2', 'AAA' // new_line('a') // 'BBB', 2)
  print *, 'PASS'
contains
  subroutine check(name, contents, nrecs)
    character(*), intent(in) :: name, contents
    integer, intent(in) :: nrecs
    character(len=32) :: buf, again
    integer :: iu, i, stat

    open (newunit=iu, file=name, status='replace', action='write', &
        form='unformatted', access='stream')
    write (iu) contents
    close (iu)

    open (newunit=iu, file=name, status='old', action='read', &
        form='formatted', access='sequential')
    do i = 1, nrecs
      read (iu, '(A)') buf
    end do
    backspace (iu, iostat=stat)
    if (stat /= 0) then
      print *, 'FAIL: ' // name // ': backspace failed'
      stop 1
    end if
    read (iu, '(A)') again
    if (again /= buf) then
      print *, 'FAIL: ' // name // ': reread mismatch'
      stop 1
    end if
    print *, 'PASS: ' // name
    close (iu, status='delete')
  end subroutine
end program

! CHECK: PASS

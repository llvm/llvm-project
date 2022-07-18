! RUN: %python %S/test_errors.py %s %flang_fc1
  integer :: unit10 = 10
  integer :: unit11 = 11
  integer, parameter :: const_stat = 6666

  integer(kind=1) :: stat1
  integer(kind=8) :: stat8

  character(len=55) :: msg

  close(unit10)
  close(unit=unit11, err=9, iomsg=msg, iostat=stat1)
  close(12, status='Keep')

  close(iostat=stat8, 11) ! nonstandard

  !ERROR: CLOSE statement must have a UNIT number specifier
  close(iostat=stat1)

  !ERROR: Duplicate UNIT specifier
  close(13, unit=14, err=9)

  !ERROR: Duplicate ERR specifier
  close(err=9, unit=15, err=9, iostat=stat8)

  !ERROR: Invalid STATUS value 'kept'
  close(status='kept', unit=16)

  !ERROR: Invalid STATUS value 'old'
  close(status='old', unit=17)

  !Ok: trailing spaces ignored
  close(status='keep ', unit=17)

  !ERROR: IOSTAT variable 'const_stat' must be definable
  close(14, iostat=const_stat)

9 continue
end

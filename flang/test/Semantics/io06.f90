! RUN: %python %S/test_errors.py %s %flang_fc1
  character(kind=1,len=100) msg1
  character(kind=2,len=200) msg2
  character, parameter :: const_msg = 'doof'
  integer(1) stat1
  integer(2) stat2
  integer(8) stat8

  open(10)

  backspace(10)
  backspace(10, iomsg=msg1, iostat=stat1, err=9)

  endfile(unit=10)
  endfile(iostat=stat2, err=9, unit=10, iomsg=msg1)

  rewind(10)
  rewind(iomsg=msg1, iostat=stat2, err=9, unit=10)

  flush(10)
  flush(iomsg=msg1, unit=10, iostat=stat8, err=9)

  wait(10)
  wait(99, id=id1, end=9, eor=9, err=9, iostat=stat1, iomsg=msg1)

  !ERROR: Duplicate UNIT specifier
  backspace(10, unit=11)

  !ERROR: Duplicate IOSTAT specifier
  endfile(iostat=stat2, err=9, unit=10, iostat=stat8, iomsg=msg1)

  !ERROR: IOMSG variable 'const_msg' is not definable
  !BECAUSE: '"d"' is not a variable or pointer
  flush(iomsg=const_msg, unit=10, iostat=stat8, err=9)

  !ERROR: REWIND statement must have a UNIT number specifier
  rewind(iostat=stat2)

  !ERROR: Duplicate ERR specifier
  flush(err=9, unit=10, &
        err=9, &
        err=9)

  !ERROR: Duplicate ID specifier
  !ERROR: WAIT statement must have a UNIT number specifier
  wait(id=id2, eor=9, id=id3)

9 continue
end

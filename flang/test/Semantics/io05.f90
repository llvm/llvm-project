! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
  character*20 c(25), cv
  character(kind=1,len=59) msg
  character, parameter :: const_round = "c'est quoi?"
  logical*2 v(5), lv
  integer*1 stat1
  integer*2 stat4
  integer*8 stat8, iv
  integer, parameter :: const_id = 1

  inquire(10)
  inquire(file='abc')
  inquire(10, pos=ipos, iomsg=msg, iostat=stat1)
  inquire(file='abc', &
      access=c(1), action=c(2), asynchronous=c(3), blank=c(4), decimal=c(5), &
      delim=c(6), direct=c(7), encoding=c(8), form=c(9), formatted=c(10), &
      name=c(11), pad=c(12), position=c(13), read=c(14), readwrite=c(15), &
      round=c(16), sequential=c(17), sign=c(18), stream=c(19), &
      unformatted=c(20), write=c(21), &
      err=9, &
      nextrec=nextrec, number=number, pos=jpos, recl=jrecl, size=jsize, &
      iomsg=msg, &
      iostat=stat4, &
      exist=v(1), named=v(2), opened=v(3), pending=v(4))
  inquire(pending=v(5), file='abc')
  inquire(10, id=id, pending=v(5))
  inquire(10, id=const_id, pending=v(5))
  inquire(10, carriagecontrol=c(1)) ! nonstandard

  ! using variable 'cv' multiple times seems to be allowed
  inquire(file='abc', &
      access=cv, action=cv, asynchronous=cv, blank=cv, decimal=cv, &
      delim=cv, direct=cv, encoding=cv, form=cv, formatted=cv, &
      name=cv, pad=cv, position=cv, read=cv, readwrite=cv, &
      round=cv, sequential=cv, sign=cv, stream=cv, &
      unformatted=cv, write=cv, &
      nextrec=iv, number=iv, pos=iv, recl=iv, size=iv, &
      exist=lv, named=lv, opened=lv, pending=lv)

  !ERROR: INQUIRE statement must have a UNIT number or FILE specifier
  inquire(err=9)

  !ERROR: If FILE appears, UNIT must not appear
  inquire(10, file='abc', blank=c(22), iostat=stat8)

  !ERROR: Duplicate FILE specifier
  inquire(file='abc', file='xyz')

  !ERROR: Duplicate FORM specifier
  inquire(form=c(1), iostat=stat1, form=c(2), file='abc')

  !ERROR: Duplicate SIGN specifier
  !ERROR: Duplicate READ specifier
  !ERROR: Duplicate WRITE specifier
  inquire(1, read=c(1), write=c(2), sign=c(3), sign=c(4), read=c(5), write=c(1))

  !ERROR: Duplicate IOMSG specifier
  inquire(10, iomsg=msg, pos=ipos, iomsg=msg)

  !ERROR: If ID appears, PENDING must also appear
  inquire(file='abc', id=id)

  !ERROR: ROUND variable 'const_round' must be definable
  inquire(file='abc', round=const_round)

9 continue
end

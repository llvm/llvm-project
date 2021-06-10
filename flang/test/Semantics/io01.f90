! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
  character(len=20) :: access = "direcT"
  character(len=20) :: access_(2) = (/"direcT", "streaM"/)
  character(len=20) :: action_(2) = (/"reaD ", "writE"/)
  character(len=20) :: asynchronous_(2) = (/"nO ", "yeS"/)
  character(len=20) :: blank_(2) = (/"nulL", "zerO"/)
  character(len=20) :: decimal_(2) = (/'commA', 'poinT'/)
  character(len=20) :: delim_(2) = (/"nonE ", "quotE"/)
  character(len=20) :: encoding_(2) = (/"defaulT", "utF-8  "/)
  character(len=20) :: form_(2) = (/"formatteD  ", "unformatteD"/)
  character(len=20) :: pad_(2) = (/"nO ", "yeS"/)
  character(len=20) :: position_(3) = (/"appenD", "asiS  ", "rewinD"/)
  character(len=20) :: round_(2) = (/"dowN", "zerO"/)
  character(len=20) :: sign_(2) = (/"pluS    ", "suppresS"/)
  character(len=20) :: status_(2) = (/"neW", "olD"/)
  character(len=20) :: convert_(2) = (/"big_endiaN", "nativE    "/)
  character(len=20) :: dispose_(2) = (/ "deletE", "keeP  "/)
  character(len=66) :: cc, msg

  integer :: new_unit
  integer :: unit10 = 10
  integer :: unit11 = 11
  integer :: n = 40
  integer, parameter :: const_new_unit = 66

  integer(kind=1) :: stat1
  integer(kind=2) :: stat2
  integer(kind=4) :: stat4
  integer(kind=8) :: stat8

  cc = 'scratch'

  open(unit10)
  open(blank='null', unit=unit10, pad='no')
  open(unit=unit11, err=3)
3 continue

  open(20, access='sequential')
  open(21, access=access, recl=n)
  open(22, access=access_(2), iostat=stat1, iomsg=msg)

  open(30, action='readwrite', asynchronous='n'//'o', blank='zero')
  open(31, action=action_(2), asynchronous=asynchronous_(2), blank=blank_(2))

  open(unit=40, decimal="comma", delim="apostrophe", encoding="utf-8")
  open(unit=41, decimal=decimal_(2), delim=delim_(2), encoding=encoding_(2))

  open(50, file='abc', status='unknown', form='formatted')
  open(51, file=access, status=status_(2), form=form_(2))

  open(newunit=new_unit, pad=pad_(2), status='scr'//'atch'//'')
  open(newunit=new_unit, pad=pad_(2), status=cc)

  open(unit=60, position='rewind', recl=(30+20/2), round='zero')
  open(position=position_(1), recl=n, round=round_(2), unit=61)

  open(unit=70, sign='suppress', &
      status='unknown', iostat=stat2)
  open(unit=70, sign=sign_(2), status=status_(2))

  open(80, convert='big_endian', dispose='delete')
  open(81, convert=convert_(2), dispose=dispose_(2))

  open(access='STREAM', 90) ! nonstandard
  open (unit=91, file='xfile', carriagecontrol='list') ! nonstandard

  !ERROR: OPEN statement must have a UNIT or NEWUNIT specifier
  !ERROR: If ACCESS='DIRECT' appears, RECL must also appear
  open(access='direct')

  !ERROR: If STATUS='STREAM' appears, RECL must not appear
  open(10, access='st'//'ream', recl=13)

  !ERROR: Duplicate NEWUNIT specifier
  !ERROR: If NEWUNIT appears, FILE or STATUS must also appear
  open(newunit=n, newunit=nn, iostat=stat4)

  !ERROR: NEWUNIT variable 'const_new_unit' must be definable
  open(newunit=const_new_unit, status=cc)

  !ERROR: Duplicate UNIT specifier
  open(unit=100, unit=100)

  !ERROR: Duplicate UNIT specifier
  open(101, delim=delim_(1), unit=102)

  !ERROR: Duplicate UNIT specifier
  open(unit=103, &
      unit=104, iostat=stat8)

  !ERROR: Duplicate UNIT specifier
  !ERROR: If ACCESS='DIRECT' appears, RECL must also appear
  open(access='dir'//'ect', 9, 9) ! nonstandard

  !ERROR: Duplicate ROUND specifier
  open(105, round=round_(1), pad='no', round='nearest')

  !ERROR: If NEWUNIT appears, UNIT must not appear
  !ERROR: If NEWUNIT appears, FILE or STATUS must also appear
  open(106, newunit=n)

  !ERROR: RECL value (-30) must be positive
  open(107, recl=40-70)

  !ERROR: RECL value (-36) must be positive
  open(108, recl=-  -  (-36)) ! nonstandard

  !ERROR: Invalid ACTION value 'reedwrite'
  open(109, access=Access, action='reedwrite', recl=77)

  !ERROR: Invalid ACTION value 'nonsense'
  open(110, action=''//'non'//'sense', recl=77)

  !ERROR: Invalid STATUS value 'cold'
  open(111, status='cold')

  !ERROR: Invalid STATUS value 'Keep'
  open(112, status='Keep')

  !ERROR: If STATUS='NEW' appears, FILE must also appear
  open(113, status='new')

  !ERROR: If STATUS='REPLACE' appears, FILE must also appear
  open(114, status='replace')

  !ERROR: If STATUS='SCRATCH' appears, FILE must not appear
  open(115, file='abc', status='scratch')

  !ERROR: If NEWUNIT appears, FILE or STATUS='SCRATCH' must also appear
  open(newunit=nn, status='old')

  !ERROR: Unimplemented CARRIAGECONTROL value 'fortran'
  open (unit=116, file='xfile', carriagecontrol='fortran') ! nonstandard

  !ERROR: Invalid CARRIAGECONTROL value 'nonsense'
  open (unit=116, file='xfile', carriagecontrol='nonsense') ! nonstandard
end

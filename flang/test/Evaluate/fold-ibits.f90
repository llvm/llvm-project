! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of IBITS exhaustively over POS/LEN ranges
module m1
  implicit integer(a-z)
  integer, parameter :: res1(*) = [((ibits(not(0),pos,len),len=0,31-pos),pos=0,31)]
  integer, parameter :: expect1(*) = [((maskr(len),len=0,31-pos),pos=0,31)]
  logical, parameter :: test1 = all(res1 == expect1)
  logical, parameter :: test2 = all([((ibits(0,pos,len),len=0,31-pos),pos=0,31)] == 0)
  integer, parameter :: mess = z'a5a55a5a'
  integer, parameter :: res3(*) = [((ibits(mess,pos,len),len=0,31-pos),pos=0,31)]
  integer, parameter :: expect3(*) = [((iand(shiftr(mess,pos),maskr(len)),len=0,31-pos),pos=0,31)]
  logical, parameter :: test3 = all(res3 == expect3)
end module

! IBITS must be folded at the kind of its first argument.  Folding it at the
! default integer kind truncates arguments of a wider kind.
module m2
  implicit integer(a-z)
  integer(1), parameter :: mess1 = int(z'5a', 1)
  integer(2), parameter :: mess2 = int(z'5a5a', 2)
  integer(8), parameter :: mess8 = int(z'5a5a5a5a5a5a5a5a', 8)
  integer(16), parameter :: mess16 = int(z'5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a', 16)
  logical, parameter :: test_kind1 = all( &
    [((ibits(mess1,pos,len),len=0,7-pos),pos=0,7)] == &
    [((iand(shiftr(mess1,pos),maskr(len,1)),len=0,7-pos),pos=0,7)])
  logical, parameter :: test_kind2 = all( &
    [((ibits(mess2,pos,len),len=0,15-pos),pos=0,15)] == &
    [((iand(shiftr(mess2,pos),maskr(len,2)),len=0,15-pos),pos=0,15)])
  logical, parameter :: test_kind8 = all( &
    [((ibits(mess8,pos,len),len=0,63-pos),pos=0,63)] == &
    [((iand(shiftr(mess8,pos),maskr(len,8)),len=0,63-pos),pos=0,63)])
  logical, parameter :: test_kind16 = all( &
    [((ibits(mess16,pos,len),len=0,127-pos),pos=0,127)] == &
    [((iand(shiftr(mess16,pos),maskr(len,16)),len=0,127-pos),pos=0,127)])
  ! Bit fields that extend past bit 31 must survive.
  logical, parameter :: test_wide8 = ibits(1234567890123_8, 0, 40) == 135056262347_8
  logical, parameter :: test_wide16 = &
    ibits(1234567890123456789_16, 8, 120) == 4822530820794753_16
  ! Folding must not narrow the first argument, which would overflow.
  integer(8), parameter :: nonarrow8 = ibits(-1_8, 0, 64)
  logical, parameter :: test_nonarrow8 = nonarrow8 == -1_8
end module

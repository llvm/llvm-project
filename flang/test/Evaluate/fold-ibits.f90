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

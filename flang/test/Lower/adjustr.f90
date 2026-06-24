! RUN: %flang_fc1 -emit-fir -o - %s | FileCheck %s

! Checking no reference to boxchar conversion is in the emitted fir.
! CHECK-NOT: (!fir.ref<!fir.char<1,4>>) -> !fir.boxchar<1>
! CHECK: %[[Const4:.*]] = arith.constant 4 : index
! CHECK: fir.emboxchar

program main
  logical         ,dimension(1):: mask = .true.
  character(len=2),dimension(1):: d1 = "a "
  character(len=4),dimension(1):: d4
  where (mask)
    d4 = adjustr(d1 // d1)
  end where
  write(6,*) "d4     =", d4
end

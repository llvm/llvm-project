! RUN: %flang -fc1 -emit-fir -cpp -DNVAR=2+1+0+0 -o - %s | FileCheck %s

! CHECK: fir.string_lit "NO"(2) : !fir.char<1,2>
! CHECK-NOT: fir.string_lit "YES"(3) : !fir.char<1,3>

program test
#if NVAR < 2
  print *, "YES"
#else
  print *, "NO"
#endif
end program


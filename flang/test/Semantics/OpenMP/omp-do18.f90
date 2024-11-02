! RUN: %flang_fc1 -fdebug-unparse-with-symbols %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -fopenmp -fdebug-unparse-with-symbols %s 2>&1 | FileCheck %s
! CHECK-NOT: do *[1-9]

program P
implicit none
integer OMP_GET_NUM_THREADS, OMP_GET_THREAD_NUM
integer NUMTHRDS, TID
integer N, CSZ, CNUM, I
parameter (N=100)
parameter (CSZ=10) 
real A(N), B(N), C(N)

do 10 I = 1, N
   A(I) = I * 1.0
10 continue

B = A
CNUM = CSZ

!$OMP PARALLEL SHARED(A,B,C,NUMTHRDS,CNUM) PRIVATE(I,TID)
TID = OMP_GET_THREAD_NUM()
if (TID .EQ. 0) then
   NUMTHRDS = OMP_GET_NUM_THREADS()
   print *, "Number of threads =", NUMTHRDS
end if
print *, "Thread", TID, " is starting..."

!$OMP DO SCHEDULE(DYNAMIC,CNUM)
do 20 I = 1, N
   C(I) = A(I) + B(I)
   write (*,100) TID, I, C(I)
20 continue
!$OMP END DO NOWAIT

print *, "Thread", TID, " done."

!$OMP END PARALLEL
100 format(" Thread", I2, ": C(", I3, ")=", F8.2)
end program P

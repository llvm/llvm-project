! RUN: %f18 -fopenmp -funparse-with-symbols %s 2>&1 | FileCheck %s
! CHECK-NOT: do *[1-9]
! CHECK: omp simd

program P
implicit none
integer N, I
parameter (N=100)
real A(N), B(N), C(N)

!$OMP SIMD
do 10 I = 1, N
   A(I) = I * 1.0
10 continue

B = A

!$OMP SIMD
do 20 I = 1, N
   C(I) = A(I) + B(I)
   write (*,100) I, C(I)
20 continue

100 format(" C(", I3, ")=", F8.2)
end program P

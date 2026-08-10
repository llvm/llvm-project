! RUN:  bbc -emit-fir %s -o - | FileCheck %s

program test_extent_from_triplet
  implicit none
  integer, parameter:: n = 3
  INTEGER a(n), b(n), i
  a = (/ 1, 2, 3 /)
  b = (/ (sum(a(1:i)), i=1, n) /)
end program

! CHECK: %{{.*}} = fir.embox %{{.*}}(%{{.*}}) [%{{.*}}] : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xi32>>

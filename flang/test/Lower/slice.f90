! RUN: bbc -o - %s | FileCheck %s
program p
  real :: a1(10,10)
  real :: a2(3)
  real :: a3(10)
  integer iv(3)
  integer k

  ! CHECK-DAG: %[[a1:.*]] = fir.address_of(@_QEa1)
  ! CHECK-DAG: %[[a2:.*]] = fir.address_of(@_QEa2)
  k = 0
  do j = 1, 10
     do i = 1, 10
        k = k + 1
        a1(i,j) = cos(real(k))
     end do
     a3(j) = sin(real(k))
  end do

  ! CHECK: %[[undef:.*]] = fir.undefined index
  ! CHECK: %[[slice:.*]] = fir.slice %{{.*}}, %[[undef]], %[[undef]], %
  ! CHECK: fir.array_coor %[[a1]](%{{.*}}) [%[[slice]]] %{{.*}}, %[[index:.*]] :
  ! CHECK: fir.array_coor %[[a2]](%{{.*}}) %[[index]] :
  a2 = a1(4, 2:10:3)

  ! CHECK: fir.address_of(@_QQcl.6D69736D617463682031)
  if (a1(4,2) .ne. a2(1)) print *, "mismatch 1", a2(1), a1(4,2)
  if (a1(4,5) .ne. a2(2)) print *, "mismatch 2", a2(2), a1(4,5)
  if (a1(4,8) .ne. a2(3)) print *, "mismatch 3", a2(3), a1(4,8)

  ! CHECK: %[[shape:.*]] = fir.shape %c10
  ! CHECK: %[[slice2:.*]] = fir.slice %c1{{.*}}, %c10{{.*}}, %c4
  ! CHECK: %[[coor:.*]] = fir.array_coor %{{.*}}(%[[shape]]) [%[[slice2]]
  ! CHECK: fir.store %{{.*}} to %[[coor]]
  a3(1:10:4) = a2

  if (a1(4,2) .ne. a3(1)) print *, "mismatch 4", a1(4,2), a3(1)
  if (a1(4,5) .ne. a3(5)) print *, "mismatch 5", a1(4,5), a3(5)
  if (a1(4,8) .ne. a3(9)) print *, "mismatch 6", a1(4,8), a3(9)

  iv = (/ 3, 1, 2 /)

  ! CHECK: %[[alloc:.*]] = fir.allocmem !fir.array<3xf32>
  ! CHECK: fir.array_coor %{{.*}}(%{{.*}}) %
  ! CHECK: fir.array_coor %[[alloc]](%{{.*}}) %
  ! CHECK: %[[slice3:.*]] = fir.slice %c1{{.*}}, %c3{{.*}}, %c1{{.*}}
  ! CHECK: fir.array_coor %{{.*}}(%{{.*}}) [%[[slice3]]
  ! CHECK: %[[tmp:.*]] = fir.array_coor %[[alloc]](%{{.*}}) %
  ! CHECK: fir.store %{{.*}} to %[[tmp]]
  ! CHECK: %[[tmp2:.*]] = fir.array_coor %[[alloc]](%{{.*}}) %
  ! CHECK: fir.load %[[tmp2]]
  ! CHECK: fir.freemem %[[alloc]] :
  a2 = a2(iv)

  if (a1(4,2) .ne. a2(2)) print *, "mismatch 7", a1(4,2), a2(2)
  if (a1(4,5) .ne. a2(3)) print *, "mismatch 8", a1(4,5), a2(3)
  if (a1(4,8) .ne. a2(1)) print *, "mismatch 9", a1(4,8), a2(1)

end program p

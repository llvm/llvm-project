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

! Slice operation on array of CHARACTER
! CHECK-LABEL: func @_QPsub
subroutine sub(a)
  character :: a(10)
  ! CHECK-DAG: %[[ten:.*]] = constant 10 : index
  ! CHECK-DAG: %[[one:.*]] = constant 1 : i64
  ! CHECK-DAG: %[[five:.*]] = constant 5 : i64
  ! CHECK-DAG: %[[two:.*]] = constant 2 : i64
  ! CHECK-DAG: %[[three:.*]] = constant 3 : index
  ! CHECK: %[[shape:.*]] = fir.shape %[[ten]] :
  ! CHECK: %[[slice:.*]] = fir.slice %[[one]], %[[five]], %[[two]] :
  ! CHECK: %[[allocmem:.*]] = fir.allocmem !fir.array<3x!fir.char<1>>
  ! CHECK: %[[shape3:.*]] = fir.shape %[[three]] :
  ! CHECK: fir.array_coor %{{.*}}(%[[shape]]) [%[[slice]]] %
  ! CHECK: fir.embox %[[allocmem]](%[[shape3]]) : (!fir.heap<!fir.array<3x!fir.char<1>>>, !fir.shape<1>) -> !fir.box<!fir.array<3x!fir.char<1>>>
  print *, "a = ", a(1:5:2)
end subroutine sub

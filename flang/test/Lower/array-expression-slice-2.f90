! RUN: bbc %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPi
subroutine i
  implicit none
  integer :: ctemp(10) = (/1,2,3,4,5,6,7,8,9,22/)
  print *, ctemp(1:10)
end subroutine i

! CHECK-LABEL: func @_QPs
subroutine s
  integer, parameter :: LONGreal = 8
  real (kind = LONGreal), dimension(-1:11) :: x = (/0,0,0,0,0,0,0,0,0,0,0,0,0/)
  real (kind = LONGreal), dimension(0:12) :: g = (/0,0,0,0,0,0,0,0,0,0,0,0,0/)
  real (kind = LONGreal) :: gs(13)
  x(1) = 4.0
  g(1) = 5.0
  gs = g(0:12:1) + x(11:(-1):(-1))
  print *, gs
  !print *, dot_product(g(0:12:1), x(11:(-1):(-1)))
end subroutine s

! CHECK-LABEL: func @_QPs2
subroutine s2
  real :: x(10)
  x = 0.0
  ! CHECK: fir.call @_FortranAioBeginExternalListOutput
  print *, x
  ! CHECK: %[[s:.*]] = fir.slice {{.*}} !fir.slice<1>
  ! CHECK: %[[p:.*]] = fir.array_coor %{{.*}} [%[[s]]] %
  ! CHECK: fir.store %{{.*}} to %[[p]] : !fir.ref<f32>
  x(1:10:3) = 2.0
  ! CHECK: fir.call @_FortranAioBeginExternalListOutput
  print *, x
end subroutine s2

! CHECK-LABEL: func @_QQmain
program main
  integer :: A(10)
  A(1) = 1
  A(2) = 2
  A(3) = 3
  print *, A
  ! CHECK: %[[A:.*]] = fir.address_of(@_QFEa)
  ! CHECK: %[[shape:.*]] = fir.shape %c10
  ! CHECK: %[[slice:.*]] = fir.slice %
  ! CHECK: fir.embox %[[A]](%[[shape]]) [%[[slice]]] :
  print*, A(1:3:1)
  call s
  call i
end program main

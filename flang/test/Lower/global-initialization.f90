! RUN: bbc %s -o - | FileCheck %s

program bar
! CHECK: fir.address_of(@[[name1:.*]]my_data)
  integer, save :: my_data = 1
  print *, my_data
contains

! CHECK-LABEL: func @_QFPfoo
subroutine foo()
! CHECK: fir.address_of(@[[name2:.*foo.*my_data]])
  integer, save :: my_data = 2
  print *, my_data + 1
end subroutine

! CHECK-LABEL: func @_QFPfoo2
subroutine foo2()
! CHECK: fir.address_of(@[[name3:.*foo2.*my_data]])
  integer, save :: my_data
  my_data = 4
  print *, my_data
end subroutine

! CHECK-LABEL: func @_QFPfoo3
subroutine foo3()
! CHECK-DAG: fir.address_of(@[[name4:.*foo3.*idata]]){{.*}}fir.array<5xi32>
! CHECK-DAG: fir.address_of(@[[name5:.*foo3.*rdata]]){{.*}}fir.array<3xf16>
! CHECK-DAG: fir.address_of(@[[name6:.*foo3.*my_data]]){{.*}}fir.array<2x4xi64>
  integer*4, dimension(5), save :: idata = (/ (i*i, i=1,5) /)
  integer*8, dimension(2, 10:13), save :: my_data = reshape((/1,2,3,4,5,6,7,8/), shape(my_data))
  real*2, dimension(7:9), save :: rdata = (/100., 99., 98./)
  print *, rdata(9)
  print *, idata(3)
  print *, my_data(1,11)
end subroutine
end program

! CHECK: fir.global internal @[[name1]]
! CHECK: fir.global internal @[[name2]]
! CHECK: fir.global internal @[[name3]]
! CHECK-DAG: fir.global internal @[[name4]]{{.*}}fir.array<5xi32>
! CHECK-DAG: fir.global internal @[[name5]]{{.*}}fir.array<3xf16>
! CHECK-DAG: fir.global internal @[[name6]]{{.*}}fir.array<2x4xi64>

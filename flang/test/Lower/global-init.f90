! RUN: bbc %s -o - | FileCheck %s

program bar
! CHECK: fir.address_of(@[[name1:.*]]my_data)
! CHECK: fir.global @[[name1]]
  integer, save :: my_data = 1
  print *, my_data
contains
subroutine foo()
! CHECK: fir.address_of(@[[name2:.*foo.*my_data]])
! CHECK: fir.global @[[name2]]
  integer, save :: my_data = 2
  print *, my_data + 1
end subroutine
subroutine foo2()
! CHECK: fir.address_of(@[[name3:.*foo2.*my_data]])
! CHECK: fir.global @[[name3]]
  integer, save :: my_data
  my_data = 4
  print *, my_data
end subroutine
subroutine foo3()
! CHECK: fir.address_of(@[[name4:.*foo3.*idata]]){{.*}}fir.array<5xi32>
! CHECK: fir.address_of(@[[name5:.*foo3.*rdata]]){{.*}}fir.array<3xf16>
! CHECK: fir.global @[[name4]]{{.*}}fir.array<5xi32>
! CHECK: fir.global @[[name5]]{{.*}}fir.array<3xf16>
  integer*4, dimension(5), save :: idata = (/ (i*i, i=1,5) /)
  real*2, dimension(7:9), save :: rdata = (/100., 99., 98./)
  print *, rdata(9)
  print *, idata(3)
end subroutine
end program

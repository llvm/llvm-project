! RUN: bbc %s -o - | FileCheck %s

program bar
! CHECK: fir.address_of(@[[name1:.*]]my_data)
! CHECK: fir.global @[[name1]]
  integer, save :: my_data = 1
  print *, my_data
call foo
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
end program

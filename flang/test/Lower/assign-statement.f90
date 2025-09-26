! RUN: bbc -emit-fir -o - %s | FileCheck %s

  ! CHECK-LABEL: func @_QQmain
  program main
  integer :: ip
  pointer :: ip

  allocate(ip)
  assign 10 to ip
  ! CHECK: fir.store %c10_i32 to %11 : !fir.ptr<i32>
  10 return
  end program main

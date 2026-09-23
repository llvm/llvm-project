! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck %s

module test_mod
! CHECK-DAG: !DIModule(scope: !{{.*}}, name: "test_mod", file: !{{.*}}, line: [[@LINE-1]])
  integer :: mod_var
contains
  subroutine test_sub()
    mod_var = 100
  end subroutine test_sub
end module test_mod

module another_mod
! CHECK-DAG: !DIModule(scope: !{{.*}}, name: "another_mod", file: !{{.*}}, line: [[@LINE-1]])
  real :: x
contains
  function get_value() result(res)
    real :: res
    res = 42.0
  end function get_value
end module another_mod

program main
  use test_mod
  use another_mod
  call test_sub()
  x = get_value()
end program main


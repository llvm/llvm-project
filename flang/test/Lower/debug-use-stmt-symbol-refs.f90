! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Test USE statement lowering to fir.use_stmt operations
! Covers: USE ONLY, USE with renames, and USE (all)

module mod1
  integer :: a = 10, b = 20, c = 30
end module mod1

module mod2
  real :: x = 1.0, y = 2.0, z = 3.0
end module mod2

module mod3
  logical :: flag = .true.
end module mod3

! Test 1: Program with USE ONLY and USE with renames
program test_main
  use mod1, only: b, c
  use mod2, renamed_y => y
  implicit none
  print *, b, c, renamed_y
end program

! Test 2: Subroutine with USE (all) and different renames
subroutine test_sub()
  use mod1
  use mod2, only: x
  use mod3, my_flag => flag
  implicit none
  print *, a, b, c, x, my_flag
end subroutine

! Test 3: Function with multiple USE patterns
function test_func() result(res)
  use mod1, only: a
  use mod2, renamed_x => x, renamed_z => z
  use mod3
  implicit none
  integer :: res
  res = a
end function

! CHECK-LABEL: func.func @_QQmain()
! CHECK-DAG: fir.use_stmt "mod1" only_symbols{{\[}}[@_QMmod1Eb, @_QMmod1Ec]]
! CHECK-DAG: fir.use_stmt "mod2" renames{{\[}}[#fir.use_rename<"renamed_y", @_QMmod2Ey>]]

! CHECK-LABEL: func.func @_QPtest_sub()
! CHECK-DAG: fir.use_stmt "mod1"{{$}}
! CHECK-DAG: fir.use_stmt "mod2" only_symbols{{\[}}[@_QMmod2Ex]]
! CHECK-DAG: fir.use_stmt "mod3" renames{{\[}}[#fir.use_rename<"my_flag", @_QMmod3Eflag>]]

! CHECK-LABEL: func.func @_QPtest_func()
! CHECK-DAG: fir.use_stmt "mod1" only_symbols{{\[}}[@_QMmod1Ea]]
! CHECK-DAG: fir.use_stmt "mod2" renames{{\[}}[#fir.use_rename<"renamed_x", @_QMmod2Ex>, #fir.use_rename<"renamed_z", @_QMmod2Ez>]]
! CHECK-DAG: fir.use_stmt "mod3"{{$}}

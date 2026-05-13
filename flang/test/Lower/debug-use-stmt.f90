! RUN: %flang_fc1 -emit-hlfir -debug-info-kind=standalone %s -o - | FileCheck %s --check-prefix=WITH_DEBUG
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s --check-prefix=NO_DEBUG
! RUN: %flang_fc1 -emit-hlfir -debug-info-kind=line-tables-only %s -o - | FileCheck %s --check-prefix=NO_DEBUG

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

! Test 4: Subroutine with USE ONLY with renames (combined)
subroutine test_only_rename()
  use mod1, only: local_b => b, c, local_a => a
  use mod2, only: local_x => x, local_y => y
  implicit none
  print *, local_a, local_b, c, local_x, local_y
end subroutine

! WITH_DEBUG-LABEL: func.func @_QQmain()
! WITH_DEBUG-DAG: fir.use_stmt "mod1" only_symbols{{\[}}[@_QMmod1Eb, @_QMmod1Ec]]
! WITH_DEBUG-DAG: fir.use_stmt "mod2" renames{{\[}}[#fir.use_rename<"renamed_y", @_QMmod2Ey>]]

! NO_DEBUG-LABEL: func.func @_QQmain()
! NO_DEBUG-NOT: fir.use_stmt

! WITH_DEBUG-LABEL: func.func @_QPtest_sub()
! WITH_DEBUG-DAG: fir.use_stmt "mod1"{{$}}
! WITH_DEBUG-DAG: fir.use_stmt "mod2" only_symbols{{\[}}[@_QMmod2Ex]]
! WITH_DEBUG-DAG: fir.use_stmt "mod3" renames{{\[}}[#fir.use_rename<"my_flag", @_QMmod3Eflag>]]

! NO_DEBUG-LABEL: func.func @_QPtest_sub()
! NO_DEBUG-NOT: fir.use_stmt

! WITH_DEBUG-LABEL: func.func @_QPtest_func()
! WITH_DEBUG-DAG: fir.use_stmt "mod1" only_symbols{{\[}}[@_QMmod1Ea]]
! WITH_DEBUG-DAG: fir.use_stmt "mod2" renames{{\[}}[#fir.use_rename<"renamed_x", @_QMmod2Ex>, #fir.use_rename<"renamed_z", @_QMmod2Ez>]]
! WITH_DEBUG-DAG: fir.use_stmt "mod3"{{$}}

! NO_DEBUG-LABEL: func.func @_QPtest_func()
! NO_DEBUG-NOT: fir.use_stmt

! WITH_DEBUG-LABEL: func.func @_QPtest_only_rename()
! WITH_DEBUG-DAG: fir.use_stmt "mod1" only_symbols{{\[}}[@_QMmod1Ec]] renames{{\[}}[#fir.use_rename<"local_b", @_QMmod1Eb>, #fir.use_rename<"local_a", @_QMmod1Ea>]]
! WITH_DEBUG-DAG: fir.use_stmt "mod2" renames{{\[}}[#fir.use_rename<"local_x", @_QMmod2Ex>, #fir.use_rename<"local_y", @_QMmod2Ey>]]

! NO_DEBUG-LABEL: func.func @_QPtest_only_rename()
! NO_DEBUG-NOT: fir.use_stmt

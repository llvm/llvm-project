! RUN: %flang_fc1 -emit-hlfir -debug-info-kind=standalone %s -o - | FileCheck %s --check-prefix=WITH_DEBUG
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s --check-prefix=NO_DEBUG
! RUN: %flang_fc1 -emit-hlfir -debug-info-kind=line-tables-only %s -o - | FileCheck %s --check-prefix=NO_DEBUG

module mod_a
  integer :: x = 10
  integer :: w = 40
  integer :: z = 50
  integer :: base = 100
end module mod_a

module mod_b
  use mod_a, only: x, renamed_w => w
  integer :: y = 20
  integer :: b_var = 200
end module mod_b

module mod_c
  use mod_b
  integer :: c_var = 300
end module mod_c

module mod_d
  use mod_c, only: y, c_var, renamed_x => x
  integer :: d_var = 400
end module mod_d

! Test 1: Function uses mod_b directly (1 level transitive)
program test_main
  use mod_b
  implicit none
  print *, x, y, renamed_w
end program

! Test 2: Function uses mod_c (2 levels transitive: mod_c -> mod_b -> mod_a)
subroutine test_sub1
  use mod_c
  implicit none
  print *, x, y, renamed_w, c_var
end subroutine

! Test 3: Function uses mod_d (3 levels transitive: mod_d -> mod_c -> mod_b -> mod_a)
function test_func() result(res)
  use mod_d
  implicit none
  integer :: res
  res = renamed_x + y + c_var + d_var
end function

! Test 4: Function uses both mod_a directly and mod_c (mixed direct and transitive)
subroutine test_sub2
  use mod_a
  use mod_c, only: c_var
  implicit none
  print *, x, w, z, base, c_var
end subroutine

! WITH_DEBUG-LABEL: func.func @_QQmain()
! WITH_DEBUG: fir.use_stmt "mod_a" only_symbols{{\[}}[@_QMmod_aEx]] renames{{\[}}[#fir.use_rename<"renamed_w", @_QMmod_aEw>]]
! WITH_DEBUG: fir.use_stmt "mod_b"

! NO_DEBUG-LABEL: func.func @_QQmain()
! NO_DEBUG-NOT: fir.use_stmt

! WITH_DEBUG-LABEL: func.func @_QPtest_sub1()
! WITH_DEBUG: fir.use_stmt "mod_a" only_symbols{{\[}}[@_QMmod_aEx]] renames{{\[}}[#fir.use_rename<"renamed_w", @_QMmod_aEw>]]
! WITH_DEBUG: fir.use_stmt "mod_b"
! WITH_DEBUG: fir.use_stmt "mod_c"

! NO_DEBUG-LABEL: func.func @_QPtest_sub1()
! NO_DEBUG-NOT: fir.use_stmt

! WITH_DEBUG-LABEL: func.func @_QPtest_func()
! WITH_DEBUG: fir.use_stmt "mod_a" only_symbols{{\[}}[@_QMmod_aEx]] renames{{\[}}[#fir.use_rename<"renamed_w", @_QMmod_aEw>]]
! WITH_DEBUG: fir.use_stmt "mod_b"
! WITH_DEBUG: fir.use_stmt "mod_c" only_symbols{{\[}}[@_QMmod_bEy, @_QMmod_cEc_var]] renames{{\[}}[#fir.use_rename<"renamed_x", @_QMmod_aEx>]]
! WITH_DEBUG: fir.use_stmt "mod_d"

! NO_DEBUG-LABEL: func.func @_QPtest_func()
! NO_DEBUG-NOT: fir.use_stmt

! WITH_DEBUG-LABEL: func.func @_QPtest_sub2()
! WITH_DEBUG: fir.use_stmt "mod_a"
! WITH_DEBUG: fir.use_stmt "mod_b"
! WITH_DEBUG: fir.use_stmt "mod_c" only_symbols{{\[}}[@_QMmod_cEc_var]]

! NO_DEBUG-LABEL: func.func @_QPtest_sub2()
! NO_DEBUG-NOT: fir.use_stmt

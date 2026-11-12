! RUN: %flang_fc1 -emit-hlfir -debug-info-kind=standalone %s -o - | FileCheck %s --check-prefix=WITH_DEBUG
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s --check-prefix=NO_DEBUG
! RUN: %flang_fc1 -emit-hlfir -debug-info-kind=line-tables-only %s -o - | FileCheck %s --check-prefix=NO_DEBUG

module mod_ma
  integer :: va = 1
end module mod_ma

module mod_mb
  integer :: vb = 2
end module mod_mb

module mod_mc
  integer :: vc = 3
end module mod_mc

! A module that uses three different modules in its specification part.
module mod_multi
  use mod_ma, only: va
  use mod_mb
  use mod_mc, renamed_vc => vc
  integer :: vx = 0
end module mod_multi

subroutine test_sub_uses_multi
  use mod_multi
  implicit none
  print *, vx
end subroutine

! WITH_DEBUG: fir.module_debug_imports "mod_multi" {
! WITH_DEBUG-NEXT: fir.use_stmt "mod_ma" only_symbols{{\[}}[@_QMmod_maEva]]
! WITH_DEBUG-NEXT: fir.use_stmt "mod_mb"{{$}}
! WITH_DEBUG-NEXT: fir.use_stmt "mod_mc" renames{{\[}}[#fir.use_rename<"renamed_vc", @_QMmod_mcEvc>]]
! WITH_DEBUG-NEXT: }

! There should be no more fir.use_stmt for mod_ma, mod_mb and mod_mc.
! WITH_DEBUG-NOT: fir.use_stmt "mod_ma"
! WITH_DEBUG-NOT: fir.use_stmt "mod_mb"
! WITH_DEBUG-NOT: fir.use_stmt "mod_mc"

! NO_DEBUG-NOT: fir.module_debug_imports

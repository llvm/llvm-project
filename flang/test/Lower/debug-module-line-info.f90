! The location of fir.module_debug_imports is what tells AddDebugInfo the line
! of the MODULE statement, so the operation is emitted for every module, even
! one with no USE statement, and only when full debug info is requested.

! RUN: %flang_fc1 -emit-hlfir -debug-info-kind=standalone %s -o - \
! RUN:   -mmlir -mlir-print-debuginfo -mmlir -mlir-print-local-scope \
! RUN:   | FileCheck %s --check-prefix=WITH_DEBUG
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s --check-prefix=NO_DEBUG
! RUN: %flang_fc1 -emit-hlfir -debug-info-kind=line-tables-only %s -o - \
! RUN:   | FileCheck %s --check-prefix=NO_DEBUG

! NO_DEBUG-NOT: fir.module_debug_imports

! WITH_DEBUG:      fir.module_debug_imports "no_use_mod" {
! WITH_DEBUG-NEXT: } loc("{{.*}}":[[@LINE+1]]:{{[0-9]+}})
module no_use_mod
  integer :: mod_var
contains
  subroutine test_sub()
    mod_var = 100
  end subroutine test_sub
end module no_use_mod

! WITH_DEBUG:      fir.module_debug_imports "using_mod" {
! WITH_DEBUG-NEXT:   fir.use_stmt "no_use_mod"
! WITH_DEBUG:      } loc("{{.*}}":[[@LINE+1]]:{{[0-9]+}})
module using_mod
  use no_use_mod
  real :: x
end module using_mod

program main
  use using_mod
  call test_sub()
  x = 1.0
end program main

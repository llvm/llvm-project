! Test the behavior of -fdiagnostics-absolute-paths: when enabled, flang prints
! absolute paths in its diagnostics instead of the (relative) path it was
! invoked with.

! REQUIRES: system-linux

! RUN: rm -rf %t.dir && mkdir -p %t.dir
! RUN: cp %s %t.dir/diagnostics-absolute-paths.f90
! RUN: cd %t.dir && not %flang_fc1 -fsyntax-only \
! RUN:     diagnostics-absolute-paths.f90 2>&1 \
! RUN:   | FileCheck %s --check-prefix=RELATIVE
! RUN: cd %t.dir && not %flang_fc1 -fsyntax-only -fdiagnostics-absolute-paths \
! RUN:     diagnostics-absolute-paths.f90 2>&1 \
! RUN:   | FileCheck %s --check-prefix=ABSOLUTE

! RELATIVE: {{^}}diagnostics-absolute-paths.f90:{{[0-9]+}}:{{[0-9]+}}:
! ABSOLUTE: {{^}}/{{.*}}diagnostics-absolute-paths.f90:{{[0-9]+}}:{{[0-9]+}}:

program test
  implicit none
  x = 1
end program

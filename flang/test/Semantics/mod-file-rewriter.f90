! RUN: rm -fr %t && mkdir %t && cd %t
! RUN: %f18 -fsyntax-only -fdebug-module-writer %s 2>&1 | FileCheck %s --check-prefix CHECK_CHANGED
! RUN: %f18 -fsyntax-only -fdebug-module-writer %s 2>&1 | FileCheck %s --check-prefix CHECK_UNCHANGED
! RUN: %f18 -fsyntax-only -fdebug-module-writer %p/Inputs/mod-file-unchanged.f90 2>&1 | FileCheck %s --check-prefix CHECK_UNCHANGED
! RUN: %f18 -fsyntax-only -fdebug-module-writer %p/Inputs/mod-file-changed.f90 2>&1 | FileCheck %s --check-prefix CHECK_CHANGED

module m
  real :: x(10)
end module m

! CHECK_CHANGED: Processing module {{.*}}.mod: module written
! CHECK_UNCHANGED: Processing module {{.*}}.mod: module unchanged, not writing

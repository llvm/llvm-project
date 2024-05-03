! REQUIRES: system-windows
!
! RUN: %clang --driver-mode=flang -### %s -Ltest 2>&1 | FileCheck %s
!
! Test that user provided paths come before the Flang runtimes and compiler-rt
! CHECK: "-libpath:test"
! CHECK: "-libpath:{{.*}}\\lib"
! CHECK: "-libpath:{{.*}}\\lib\\clang\\{{[0-9]+}}\\lib\\windows"

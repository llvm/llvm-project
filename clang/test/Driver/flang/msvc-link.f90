! RUN: %clang --driver-mode=flang --target=x86_64-pc-windows-msvc -### %s -Ltest 2>&1 | FileCheck %s
!
! Test that user provided paths come before the Flang runtimes
! CHECK: "-libpath:test"
! CHECK: "-libpath:{{.*(\\|/)}}lib"

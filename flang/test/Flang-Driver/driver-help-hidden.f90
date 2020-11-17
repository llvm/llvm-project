! REQUIRES: new-flang-driver

!--------------------------
! FLANG DRIVER (flang-new)
!--------------------------
! RUN: %flang-new --help-hidden 2>&1 | FileCheck %s
! RUN: not %flang-new  -help-hidden 2>&1 | FileCheck %s --check-prefix=ERROR-FLANG

!----------------------------------------
! FLANG FRONTEND DRIVER (flang-new -fc1)
!----------------------------------------
! RUN: not %flang-new -fc1 --help-hidden 2>&1 | FileCheck %s --check-prefix=ERROR-FLANG-FC1
! RUN: not %flang-new -fc1  -help-hidden 2>&1 | FileCheck %s --check-prefix=ERROR-FLANG-FC1

!----------------------------------------------------
! EXPECTED OUTPUT FOR FLANG DRIVER (flang-new)
!----------------------------------------------------
! CHECK:USAGE: flang-new
! CHECK-EMPTY:
! CHECK-NEXT:OPTIONS:
! CHECK-NEXT: -###      Print (but do not run) the commands to run for this compilation
! CHECK-NEXT: -E        Only run the preprocessor
! CHECK-NEXT: -fcolor-diagnostics    Enable colors in diagnostics
! CHECK-NEXT: -fno-color-diagnostics Disable colors in diagnostics
! CHECK-NEXT: -help     Display available options
! CHECK-NEXT: -o <file> Write output to <file>
! CHECK-NEXT: -test-io  Run the InputOuputTest action. Use for development and testing only.
! CHECK-NEXT: --version Print version information

!-------------------------------------------------------------
! EXPECTED OUTPUT FOR FLANG DRIVER (flang-new)
!-------------------------------------------------------------
! ERROR-FLANG: error: unknown argument '-help-hidden'; did you mean '--help-hidden'?

!-------------------------------------------------------------
! EXPECTED OUTPUT FOR FLANG FRONTEND DRIVER (flang-new -fc1)
!-------------------------------------------------------------
! Frontend driver -help-hidden is not supported
! ERROR-FLANG-FC1: error: unknown argument: '{{.*}}'


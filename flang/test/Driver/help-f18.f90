! REQUIRES: old-flang-driver

! RUN: %flang -h 2>&1 | FileCheck %s
! RUN: %flang -help 2>&1 | FileCheck %s
! RUN: %flang --help 2>&1 | FileCheck %s
! RUN: %flang -? 2>&1 | FileCheck %s

! CHECK: f18: LLVM Fortran compiler

! CHECK:   -help                print this again
! CHECK: Unrecognised options are passed through to the external compiler
! CHECK: set by F18_FC (see defaults).

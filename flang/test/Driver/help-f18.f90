! RUN: %f18 -h 2>&1 | FileCheck %s
! RUN: %f18 -help 2>&1 | FileCheck %s
! RUN: %f18 --help 2>&1 | FileCheck %s
! RUN: %f18 -? 2>&1 | FileCheck %s

! CHECK: f18: LLVM Fortran compiler

! CHECK:   -help                print this again
! CHECK: Unrecognised options are passed through to the external compiler
! CHECK: set by F18_FC (see defaults).

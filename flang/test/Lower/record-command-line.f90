! The actual command line is recorded by the frontend and passed on to FC1 as
! the argument to -record-command-line, so in this test, we just match against
! some string with spaces that mimics what a hypothetical command line.

! RUN: %flang_fc1 -record-command-line "exec -o infile" %s -emit-fir -o - | FileCheck %s

! CHECK: module attributes {
! CHECK-SAME: llvm.commandline = "exec -o infile"


// RUN: env CINDEXTEST_EDITING=1 c-index-test -test-load-source-reparse 5 local "-remap-file=%S/Inputs/preamble-reparse-1.c,%S/Inputs/preamble-reparse-2.c" %S/Inputs/preamble-reparse-1.c | FileCheck %s
// CHECK: preamble-reparse-1.c:1:5: VarDecl=x:1:5 Extent=[1:1 - 1:6]

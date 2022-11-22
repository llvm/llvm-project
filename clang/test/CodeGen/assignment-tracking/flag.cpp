// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -debug-info-kind=standalone -O0 \
// RUN:     -emit-llvm  -fexperimental-assignment-tracking %s -o -               \
// RUN: | FileCheck %s --check-prefixes=FLAG
// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -debug-info-kind=standalone -O0 \
// RUN:     -emit-llvm  %s -o -                                                  \
// RUN: | FileCheck %s --check-prefixes=NO-FLAG

// Check some assignment-tracking stuff appears in the output when the flag
// -fexperimental-assignment-tracking is used, and that it doesn't when
// the flag is not used (default behaviour: no assignment tracking).

// FLAG: DIAssignID
// FLAG: dbg.assign

// NO-FLAG-NOT: DIAssignID
// NO-FLAG-NOT: dbg.assign

void fun(int a) {}

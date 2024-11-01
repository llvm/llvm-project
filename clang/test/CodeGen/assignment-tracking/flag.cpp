//// Explicitly enabled:
// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -debug-info-kind=standalone -O0 \
// RUN:     -emit-llvm  -fexperimental-assignment-tracking %s -o -               \
// RUN: | FileCheck %s --check-prefixes=ENABLE
//// Disabled by default:
// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -debug-info-kind=standalone -O0 \
// RUN:     -emit-llvm  %s -o -                                                  \
// RUN: | FileCheck %s --check-prefixes=DISABLE
//// Explicitly disabled:
// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -debug-info-kind=standalone -O0 \
// RUN:     -emit-llvm  %s -o - -fno-experimental-assignment-tracking            \
// RUN: | FileCheck %s --check-prefixes=DISABLE

// Check some assignment-tracking stuff appears in the output when the flag
// -fexperimental-assignment-tracking is used, that it doesn't when
// -fno-experimental-assignment-tracking is used or neither flag is specified.

// ENABLE: DIAssignID
// ENABLE: dbg.assign

// DISABLE-NOT: DIAssignID
// DISABLE-NOT: dbg.assign

void fun(int a) {}

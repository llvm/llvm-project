// issue 62985
// When 3rd-party header files are included as system headers, their overloaded
// new and delete operators are also considered as the std ones. However, those
// overloaded operator functions will also be inlined. This makes the same
// symbolic memory marked as released twice, which leads to a false uaf alarm.
//
// The first run, include as system header. False uaf report before fix.
//
// RUN: %clang_analyze_cc1 %s \
// RUN: -analyzer-checker=core,cplusplus.NewDelete,debug.ExprInspection \
// RUN:   -isystem %S/Inputs/ 2>&1 | \
// RUN:   FileCheck %s
//
// The second run, include as user header. Should always silent.
//
// RUN: %clang_analyze_cc1 %s \
// RUN: -analyzer-checker=core,cplusplus.NewDelete,debug.ExprInspection \
// RUN:   -I %S/Inputs/ 2>&1 | \
// RUN:   FileCheck %s

#include "overloaded-delete-in-header.h"

void deleteInHeader(DeleteInHeader *p) { delete p; }

// CHECK-NOT: Released

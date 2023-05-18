// RUN: %clang_cc1 -x c-header %s -emit-pch -o %t 2>&1 | FileCheck %s
// CHECK: precompiled header uses __DATE__ or __TIME__
const char *p = __DATE__;

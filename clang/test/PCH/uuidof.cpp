// RUN: %clang_cc1 -fms-extensions -x c++-header -emit-pch -o %t %s
// RUN: %clang_cc1 -fms-extensions -include-pch %t -fsyntax-only %s -emit-llvm -o - | FileCheck %s

#ifndef HEADER
#define HEADER
struct _GUID {};
const _GUID &x = __uuidof(0);
// CHECK-DAG: @_GUID_00000000_0000_0000_0000_000000000000
#endif

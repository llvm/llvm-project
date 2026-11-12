// RUN: %clang_cc1 -fsyntax-only -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

#pragma pack (push, 1)
#pragma pack()
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:14-[[@LINE-1]]:14}:"pop"

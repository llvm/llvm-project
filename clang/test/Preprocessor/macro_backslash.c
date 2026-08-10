// RUN: %clang_cc1 -E %s -Dfoo='bar\' | FileCheck %s
// CHECK: TTA bar\ TTB
TTA foo TTB

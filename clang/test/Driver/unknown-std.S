// RUN: %clang -std=c++11 %s -E -o /dev/null 2>&1 | FileCheck %s --allow-empty
// CHECK-NOT: error

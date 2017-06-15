#include "Inputs/MultiFileTUHeader.h"

Foo::Foo() {}  // CHECK: rename "{{.*}}MultiFileTU.cpp" [[@LINE]]:1 -> [[@LINE]]:4
// CHECK: rename "{{.*}}MultiFileTU.cpp" [[@LINE-1]]:6 -> [[@LINE-1]]:9

void Foo::method() { } // CHECK: rename "{{.*}}MultiFileTU.cpp" [[@LINE]]:6 -> [[@LINE]]:9

// RUN: clang-refactor-test rename-initiate -at=%s:3:6 -at=%s:6:6 -new-name=Bar %s | FileCheck %s
// RUN: clang-refactor-test rename-initiate -at=%s:3:6 -at=%s:6:6 -new-name=Bar %s | FileCheck %S/Inputs/MultiFileTUHeader.h

// Test that -print-diagnostic-options prints warning groups and disablers

// RUN: %clang -print-diagnostic-options | FileCheck %s

// CHECK:  -W
// CHECK:  -Wno-
// CHECK:  -W#pragma-messages
// CHECK:  -Wno-#pragma-messages
// CHECK:  -W#warnings
// CHECK:  -Wabi
// CHECK:  -Wno-abi
// CHECK:  -Wall
// CHECK:  -Wno-all

// Check that the default analyzer checkers for PS4 are:
//   core
//   cplusplus
//   deadcode
//   nullability
//   unix
// Excluding:
//   unix.API
//   unix.Vfork

// Check for expected checkers
// RUN: %clang -target x86_64-scei-ps4 --analyze %s -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PS4-POS-CHECKERS
//
// Negative check for unexpected checkers
// RUN: %clang -target x86_64-scei-ps4 --analyze %s -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PS4-NEG-CHECKERS
//
// Check for all unix checkers except API and Vfork
// RUN: %clang -target x86_64-scei-ps4 --analyze %s -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PS4-UNIX-CHECKERS

// CHECK-PS4-POS-CHECKERS-DAG: analyzer-checker=core
// CHECK-PS4-POS-CHECKERS-DAG: analyzer-checker=cplusplus
// CHECK-PS4-POS-CHECKERS-DAG: analyzer-checker=deadcode
// CHECK-PS4-POS-CHECKERS-DAG: analyzer-checker=nullability
//
// CHECK-PS4-NEG-CHECKERS-NOT: analyzer-checker={{osx|security}}
//
// CHECK-PS4-UNIX-CHECKERS: analyzer-checker=unix
// CHECK-PS4-UNIX-CHECKERS-DAG: analyzer-disable-checker=unix.API
// CHECK-PS4-UNIX-CHECKERS-DAG: analyzer-disable-checker=unix.Vfork
// CHECK-PS4-UNIX-CHECKERS-NOT: analyzer-checker=unix.{{API|Vfork}}

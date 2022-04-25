// Check that the default analyzer checkers for PS4/PS5 are:
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
// RUN:   | FileCheck %s --check-prefix=CHECK-POS-CHECKERS
// RUN: %clang -target x86_64-sie-ps5 --analyze %s -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-POS-CHECKERS
//
// Negative check for unexpected checkers
// RUN: %clang -target x86_64-scei-ps4 --analyze %s -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NEG-CHECKERS
// RUN: %clang -target x86_64-scei-ps4 --analyze %s -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NEG-CHECKERS
//
// Check for all unix checkers except API and Vfork
// RUN: %clang -target x86_64-scei-ps4 --analyze %s -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-UNIX-CHECKERS
// RUN: %clang -target x86_64-scei-ps4 --analyze %s -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-UNIX-CHECKERS

// CHECK-POS-CHECKERS-DAG: analyzer-checker=core
// CHECK-POS-CHECKERS-DAG: analyzer-checker=cplusplus
// CHECK-POS-CHECKERS-DAG: analyzer-checker=deadcode
// CHECK-POS-CHECKERS-DAG: analyzer-checker=nullability
//
// CHECK-NEG-CHECKERS-NOT: analyzer-checker={{osx|security}}
//
// CHECK-UNIX-CHECKERS: analyzer-checker=unix
// CHECK-UNIX-CHECKERS-DAG: analyzer-disable-checker=unix.API
// CHECK-UNIX-CHECKERS-DAG: analyzer-disable-checker=unix.Vfork
// CHECK-UNIX-CHECKERS-NOT: analyzer-checker=unix.{{API|Vfork}}

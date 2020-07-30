// REQUIRES: system-darwin

// Always error about undefined 'TARGET_OS_*' macros on Darwin.
// RUN: %clang -### %s 2>&1 | FileCheck %s

// CHECK-DAG: "-Wundef-prefix=TARGET_OS_"
// CHECK-DAG: "-Werror=undef-prefix"

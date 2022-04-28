// PS4/PS5 miscellaneous toolchain defaults.

// RUN: %clang -c %s -### -target x86_64-scei-ps4 2>&1 | FileCheck %s
// RUN: %clang -c %s -### -target x86_64-sie-ps5 2>&1 | FileCheck %s
// CHECK-DAG: "-ffunction-sections"
// CHECK-DAG: "-fdata-sections"
// CHECK-DAG: "-fdeclspec"

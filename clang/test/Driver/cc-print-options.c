// RUN: rm -f %t.log
// RUN: env CC_PRINT_OPTIONS=1 \
// RUN:     CC_PRINT_OPTIONS_FILE=%t.log \
// RUN: %clang -S -o %t.s %s
// RUN: FileCheck %s < %t.log

// RUN: rm -f %t.log
// RUN: env CC_PRINT_OPTIONS=1 \
// RUN:     CC_PRINT_OPTIONS_FILE=%t.log \
// RUN: %clang -fdriver-only -o %t.s %s
// RUN: FileCheck %s < %t.log

// CHECK: [Logging clang options]
// CHECK: {{.*}} "-cc1"

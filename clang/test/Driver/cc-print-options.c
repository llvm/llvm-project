// RUN: env CC_PRINT_OPTIONS=1 \
// RUN:     CC_PRINT_OPTIONS_FILE=%t.log \
// RUN: %clang -no-canonical-prefixes -S -o %t.s %s
// RUN: FileCheck %s < %t.log

// CHECK: [Logging clang options]
// CHECK: {{.*}}clang{{.*}}"-S"


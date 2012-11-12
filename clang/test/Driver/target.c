// RUN: %clang -no-canonical-prefixes -target unknown-unknown-unknown -c %s \
// RUN:   -o %t.o -### 2>&1 | FileCheck %s
//
// Ensure we get a crazy triple here as we asked for one.
// CHECK: Target: unknown-unknown-unknown
//
// Also, ensure we don't blindly hand our target selection logic down to GCC.
// CHECK: "{{.*gcc(\.[Ee][Xx][Ee])?}}"
// CHECK-NOT: "-target"
// CHECK-NOT: "unknown-unknown-unknown"
// CHECK: "-x" "assembler"

// Check that -save-temps doesn't overwrite a .S input with its preprocessed
// output, whose name ends in .s, on a case-insensitive file system.

// REQUIRES: case-insensitive-filesystem

// RUN: rm -rf %t && mkdir -p %t && touch %t/t.S
// RUN: cd %t && %clang --target=x86_64-linux-gnu -c -save-temps t.S -### 2>&1 \
// RUN:   | FileCheck %s
// CHECK: "-E"
// CHECK-NOT: "-o" "t.s"
// CHECK-SAME: "-o" "{{[^"]*}}t-{{[^"]*}}.s"

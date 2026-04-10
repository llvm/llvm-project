// RUN: rm -rf %t-mcp
// RUN: mkdir -p %t-mcp
// REQUIRES: x86-registered-target

// Check that -module-file-deps=direct only includes directly imported PCMs,
// not transitive ones.

// RUN: %clang_cc1 -isysroot %S/Inputs/System -triple x86_64-apple-darwin10 \
// RUN:   -module-file-deps=direct -dependency-file %t.d -MT %s.o \
// RUN:   -I %S/Inputs -fmodules -fimplicit-module-maps -fdisable-module-hash \
// RUN:   -fmodules-cache-path=%t-mcp -fsyntax-only %s
// RUN: FileCheck %s < %t.d

// The directly imported module should appear.
// CHECK: diamond_bottom.pcm

// Transitive dependencies should not appear.
// CHECK-NOT: diamond_left.pcm
// CHECK-NOT: diamond_right.pcm
// CHECK-NOT: diamond_top.pcm

// -module-file-deps=all includes all transitive PCMs to show they all are used.
// RUN: %clang_cc1 -isysroot %S/Inputs/System -triple x86_64-apple-darwin10 \
// RUN:   -module-file-deps=all -dependency-file %t-all.d -MT %s.o \
// RUN:   -I %S/Inputs -fmodules -fimplicit-module-maps -fdisable-module-hash \
// RUN:   -fmodules-cache-path=%t-mcp -fsyntax-only %s
// RUN: FileCheck --check-prefix=ALL %s < %t-all.d

// ALL-DAG: diamond_bottom.pcm
// ALL-DAG: diamond_left.pcm
// ALL-DAG: diamond_right.pcm
// ALL-DAG: diamond_top.pcm

@import diamond_bottom;

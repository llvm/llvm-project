// RUN: env RC_DEBUG_OPTIONS=1 %clang -target i386-apple-darwin9 -g -Os %s  -emit-llvm -S -o - | FileCheck %s
// <rdar://problem/7256886>
// RUN: touch %t.s
// RUN: env RC_DEBUG_OPTIONS=1 %clang -### -target i386-apple-darwin9 -c -g %t.s 2>&1 | FileCheck -check-prefix=S %s

// CHECK: !0 = metadata !{
// CHECK: -g -Os
// CHECK: -mmacosx-version-min=10.5.0
// CHECK: [ DW_TAG_compile_unit ]

int x;

// S: "-dwarf-debug-flags"

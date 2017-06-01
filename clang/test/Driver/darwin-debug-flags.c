// RUN: env RC_DEBUG_OPTIONS=1 %clang -target i386-apple-darwin9 -I "path with \spaces" -g -Os %s  -emit-llvm -S -o - | FileCheck %s
// <rdar://problem/7256886>
// RUN: touch %t.s
// RUN: env RC_DEBUG_OPTIONS=1 %clang -### -target i386-apple-darwin9 -c -g %t.s 2>&1 | FileCheck -check-prefix=S %s
// <rdar://problem/12955296>
// RUN: %clang -### -target i386-apple-darwin9 -c -g %t.s 2>&1 | FileCheck -check-prefix=P %s

// CHECK: distinct !DICompileUnit(
// CHECK-SAME:                flags:
// CHECK-SAME:                -I path\5C with\5C \5C\5Cspaces
// CHECK-SAME:                -g -Os
// CHECK-SAME:                -mmacosx-version-min=10.5.0

int x;

// S: "-dwarf-debug-flags"

// P: "-dwarf-debug-producer"

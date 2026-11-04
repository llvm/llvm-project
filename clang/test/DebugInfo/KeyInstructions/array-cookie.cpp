// RUN: %clang_cc1 -triple x86_64-linux-gnu -gkey-instructions %s -debug-info-kind=line-tables-only -gno-column-info -emit-llvm -o - \
// RUN: | FileCheck %s

// Array cookie store doesn't need to be a key instruction.

struct a { char c; ~a(); };
void b() { new a[2]; }

// CHECK:      %call = call {{.*}}ptr @_Znam(i64 noundef 10)
// CHECK-NEXT: store i64 2, ptr %call, align 8, !dbg [[DBG:!.*]]

// CHECK: [[DBG]] = !DILocation(line: 7, scope: ![[#]])

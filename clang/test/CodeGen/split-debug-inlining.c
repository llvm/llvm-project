// RUN: %clang_cc1 -debug-info-kind=limited -S -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -debug-info-kind=limited -fsplit-dwarf-inlining -S -emit-llvm -o - %s | FileCheck --check-prefix=ABSENT %s
void f(void) {}
// Verify that disabling split debug inlining info is propagated to the debug
// info metadata.
// CHECK: !DICompileUnit({{.*}}, splitDebugInlining: false
// ABSENT-NOT: splitDebugInlining

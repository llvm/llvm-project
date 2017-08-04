// RUN: %clang_cc1 -triple thumbv7-apple-ios5  -target-feature +long-calls -emit-llvm -o - %s | FileCheck -check-prefix=LONGCALL %s
// RUN: %clang_cc1 -triple thumbv7-apple-ios5 -emit-llvm -o - %s | FileCheck -check-prefix=NOLONGCALL %s

// LONGCALL: attributes #0 = { {{.*}} "target-features"="+long-calls,+thumb-mode"
// NOLONGCALL-NOT: attributes #0 = { {{.*}} "target-features"="+long-calls,+thumb-mode"

int foo1(int a) { return a; }

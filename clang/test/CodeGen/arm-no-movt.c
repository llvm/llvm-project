// RUN: %clang_cc1 -triple thumbv7-apple-ios5  -target-feature +no-movt -emit-llvm -o - %s | FileCheck -check-prefix=NO-MOVT %s
// RUN: %clang_cc1 -triple thumbv7-apple-ios5 -emit-llvm -o - %s | FileCheck -check-prefix=MOVT %s

// NO-MOVT: attributes #0 = { {{.*}} "target-features"="+armv7-a,+no-movt,+thumb-mode"
// MOVT-NOT: attributes #0 = { {{.*}} "target-features"="+armv7-a,+no-movt,+thumb-mode"

int foo1(int a) { return a; }

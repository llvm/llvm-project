// Verify that -fsanitize-compilation-dir makes absolute filenames relative
// in UBSan check metadata.
//
// We use -fsanitize=unreachable (one of the checks under -fsanitize=undefined)
// rather than -fsanitize=undefined because the latter is a driver-level umbrella
// flag not accepted by cc1. Any individual sanitizer check exercises the same
// EmitCheckSourceLocation path.

// RUN: mkdir -p %t.dir/sub
// RUN: cp %s %t.dir/sub/test.c
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -emit-llvm -fsanitize=unreachable %t.dir/sub/test.c -o - | FileCheck -check-prefix=CHECK-DEFAULT %s
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -emit-llvm -fsanitize=unreachable -fsanitize-compilation-dir=%t.dir/ %t.dir/sub/test.c -o - | FileCheck -check-prefix=CHECK-STRIPPED %s
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -emit-llvm -fsanitize=unreachable -fsanitize-compilation-dir=%t.dir %t.dir/sub/test.c -o - | FileCheck -check-prefix=CHECK-STRIPPED %s
// RUN: cd %t.dir && %clang_cc1 -triple=x86_64-linux-gnu -emit-llvm -fsanitize=unreachable -fsanitize-compilation-dir=. %t.dir/sub/test.c -o - | FileCheck -check-prefix=CHECK-STRIPPED %s
// RUN: cd %t.dir/sub && %clang_cc1 -triple=x86_64-linux-gnu -emit-llvm -fsanitize=unreachable -fsanitize-compilation-dir=.. %t.dir/sub/test.c -o - | FileCheck -check-prefix=CHECK-STRIPPED %s

// Verify that a partial prefix match does not strip.
// RUN: mkdir -p %t.directory
// RUN: cp %s %t.directory/test.c
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -emit-llvm -fsanitize=unreachable -fsanitize-compilation-dir=%t.dir %t.directory/test.c -o - | FileCheck -check-prefix=CHECK-NO-FALSE-MATCH %s

// CHECK-DEFAULT: @{{.*}} = private unnamed_addr constant [{{.*}} x i8] c"{{.+}}test.c\00"
// CHECK-STRIPPED: @{{.*}} = private unnamed_addr constant [{{.*}} x i8] c"sub{{.}}test.c\00"
// CHECK-NO-FALSE-MATCH: @{{.*}} = private unnamed_addr constant [{{.*}} x i8] c"{{.+}}directory{{.}}test.c\00"
void f(void) {
  __builtin_unreachable();
}

// Verify that -fsanitize-compilation-dir makes absolute filenames relative
// in the ASan module name embedded in instrumented code.

// RUN: mkdir -p %t.dir/sub
// RUN: cp %s %t.dir/sub/test.c
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -emit-llvm -fsanitize=address %t.dir/sub/test.c -o - | FileCheck -check-prefix=CHECK-DEFAULT %s
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -emit-llvm -fsanitize=address -fsanitize-compilation-dir=%t.dir/ %t.dir/sub/test.c -o - | FileCheck -check-prefix=CHECK-STRIPPED %s
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -emit-llvm -fsanitize=address -fsanitize-compilation-dir=%t.dir %t.dir/sub/test.c -o - | FileCheck -check-prefix=CHECK-STRIPPED %s
// RUN: cd %t.dir && %clang_cc1 -triple=x86_64-linux-gnu -emit-llvm -fsanitize=address -fsanitize-compilation-dir=. %t.dir/sub/test.c -o - | FileCheck -check-prefix=CHECK-STRIPPED %s
// RUN: cd %t.dir/sub && %clang_cc1 -triple=x86_64-linux-gnu -emit-llvm -fsanitize=address -fsanitize-compilation-dir=.. %t.dir/sub/test.c -o - | FileCheck -check-prefix=CHECK-STRIPPED %s

// Verify that a partial prefix match does not strip (e.g. /tmp/dir vs /tmp/directory).
// RUN: mkdir -p %t.directory
// RUN: cp %s %t.directory/test.c
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -emit-llvm -fsanitize=address -fsanitize-compilation-dir=%t.dir %t.directory/test.c -o - | FileCheck -check-prefix=CHECK-NO-FALSE-MATCH %s

// CHECK-DEFAULT: @___asan_gen_module = private constant [{{.*}} x i8] c"{{.+}}test.c\00"
// CHECK-STRIPPED: @___asan_gen_module = private constant [{{.*}} x i8] c"sub{{.}}test.c\00"
// CHECK-NO-FALSE-MATCH: @___asan_gen_module = private constant [{{.*}} x i8] c"{{.+}}directory{{.}}test.c\00"
int x;

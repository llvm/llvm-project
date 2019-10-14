// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -emit-llvm -fsanitize=unreachable -o - | FileCheck %s -check-prefix=REGULAR -check-prefix=CHECK
// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -emit-llvm -fsanitize=unreachable -o - -fsanitize-undefined-strip-path-components=0 | FileCheck %s -check-prefix=REGULAR -check-prefix=CHECK
// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -emit-llvm -fsanitize=unreachable -o - -fsanitize-undefined-strip-path-components=2 | FileCheck %s -check-prefix=REMOVE-FIRST-TWO -check-prefix=CHECK

// Try to strip too much:
// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -emit-llvm -fsanitize=unreachable -o - -fsanitize-undefined-strip-path-components=-99999 | FileCheck %s -check-prefix=REGULAR
// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -emit-llvm -fsanitize=unreachable -o - -fsanitize-undefined-strip-path-components=99999 | FileCheck %s -check-prefix=LAST-ONLY

// Check stripping from the file name
// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -emit-llvm -fsanitize=unreachable -o - -fsanitize-undefined-strip-path-components=-2 | FileCheck %s -check-prefix=LAST-TWO
// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -emit-llvm -fsanitize=unreachable -o - -fsanitize-undefined-strip-path-components=-1 | FileCheck %s -check-prefix=LAST-ONLY

// REGULAR: @[[SRC:[0-9.a-zA-Z_]+]] =      private unnamed_addr constant [{{.*}} x i8] c"{{.*test(.|\\\\)CodeGen(.|\\\\)ubsan-strip-path-components\.cpp}}\00", align 1

// First path component: "/" or "$drive_letter:", then a name, or '\\' on Windows
// REMOVE-FIRST-TWO: @[[STR:[0-9.a-zA-Z_]+]] = private unnamed_addr constant [{{.*}} x i8] c"{{(.:|/)([^\\/]*(/|\\\\))}}[[REST:.*ubsan-strip-path-components\.cpp]]\00", align 1
// REMOVE-FIRST-TWO: @[[SRC:[0-9.a-zA-Z_]+]] = private unnamed_addr constant [{{.*}} x i8] c"[[REST]]\00", align 1

// LAST-TWO: @[[SRC:[0-9.a-zA-Z_]+]] =     private unnamed_addr constant [{{.*}} x i8] c"CodeGen{{/|\\\\}}ubsan-strip-path-components.cpp\00", align 1
// LAST-ONLY: @[[SRC:[0-9.a-zA-Z_]+]] =    private unnamed_addr constant [{{.*}} x i8] c"ubsan-strip-path-components.cpp\00", align 1

// CHECK: @[[STATIC_DATA:[0-9.a-zA-Z_]+]] = private unnamed_addr global { { [{{.*}} x i8]*, i32, i32 } } { { [{{.*}} x i8]*, i32, i32 } { [{{.*}} x i8]* @[[SRC]], i32 [[@LINE+6]], i32 3 } }
void g(const char *);
void f() {
  // CHECK-LABEL: @_Z1fv(
  g(__FILE__);
  // CHECK: call void @__ubsan_handle_builtin_unreachable(i8* bitcast ({ { [{{.*}} x i8]*, i32, i32 } }* @[[STATIC_DATA]] to i8*)) {{.*}}, !nosanitize
  __builtin_unreachable();
}

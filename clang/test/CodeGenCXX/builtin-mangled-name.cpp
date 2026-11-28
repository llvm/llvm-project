// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,ITANIUM
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -std=c++17 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,MS

// ITANIUM: @[[INT_NAME:[^ ]+]] = private unnamed_addr constant [6 x i8] c"_ZTSi\00"
// MS: @[[INT_NAME:[^ ]+]] = linkonce_odr dso_local unnamed_addr constant [4 x i8] c"?AH\00"
// CHECK: @Global = {{.*}}global ptr @[[INT_NAME]]
extern "C" {
const char *Global = __builtin_mangled_name(int);
}

// CHECK-LABEL: define {{.*}}ptr @name()
// CHECK: ret ptr @[[INT_NAME]]
extern "C" const char *name() {
  return __builtin_mangled_name(int);
}

// CHECK-LABEL: define {{.*}}i8 @first()
// CHECK: load i8, ptr @[[INT_NAME]]
extern "C" char first() {
  return __builtin_mangled_name(int)[0];
}

// CHECK-LABEL: define {{.*}}ptr @local()
// CHECK: store ptr @[[INT_NAME]], ptr %[[RESULT:.*]],
// CHECK: load ptr, ptr %[[RESULT]],
extern "C" const char *local() {
  const char *Result = __builtin_mangled_name(int);
  return Result;
}

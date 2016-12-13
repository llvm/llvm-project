// RUN: %clang_cc1 -x objective-c -emit-llvm -triple x86_64-apple-macosx10.10.0 -fsanitize=bool %s -o - | FileCheck %s -check-prefixes=SHARED,OBJC
// RUN: %clang_cc1 -x objective-c++ -emit-llvm -triple x86_64-apple-macosx10.10.0 -fsanitize=bool %s -o - | FileCheck %s -check-prefixes=SHARED,OBJC
// RUN: %clang_cc1 -x c -emit-llvm -triple x86_64-apple-macosx10.10.0 -fsanitize=bool %s -o - | FileCheck %s -check-prefixes=SHARED,C

typedef signed char BOOL;

// SHARED-LABEL: f1
BOOL f1() {
  // OBJC: call void @__ubsan_handle_load_invalid_value
  // C-NOT: call void @__ubsan_handle_load_invalid_value
  BOOL a = 2;
  return a + 1;
}

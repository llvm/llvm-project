// RUN: %clang_cc1 -triple x86_64-windows-gnu -DDLLEXPORT='__attribute__((dllexport))' -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-ibm-aix -mdefault-visibility-export-mapping=all -DDLLEXPORT= -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s
//
// The `.inline` clone of an inline builtin has internal linkage, so it must
// not get a DLL storage class even if the declaration is dllexport.

typedef __SIZE_TYPE__ size_t;

size_t strlen(const char *s);

extern inline __attribute__((always_inline)) __attribute__((gnu_inline))
DLLEXPORT size_t strlen(const char *s) {
  return 0;
}

size_t call_strlen(const char *s) {
  return strlen(s);
}

// CHECK-LABEL: define {{.*}}i64 @call_strlen(
// CHECK: call i64 @strlen.inline(
//
// CHECK: declare {{.*}}i64 @strlen(
//
// CHECK-LABEL: define internal i64 @strlen.inline(

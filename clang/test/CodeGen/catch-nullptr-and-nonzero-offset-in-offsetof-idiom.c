// RUN: %clang_cc1 -x c   -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s
// RUN: %clang_cc1 -x c   -fsanitize=pointer-overflow -fno-sanitize-recover=pointer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s

// RUN: %clang_cc1 -x c++ -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s
// RUN: %clang_cc1 -x c++ -fsanitize=pointer-overflow -fno-sanitize-recover=pointer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct S {
  int x, y;
};

// CHECK-LABEL: define i64 @{{.*}}get_offset_of_y_naively{{.*}}(
uintptr_t get_offset_of_y_naively() {
  // CHECK: [[ENTRY:.*]]:
  // CHECK-NEXT:   ret i64 ptrtoint (i32* getelementptr (i32, i32* null, i32 1) to i64)
  // CHECK-NEXT: }
  return ((uintptr_t)(&(((struct S *)0)->y)));
}

// CHECK-LABEL: define i64 @{{.*}}get_offset_of_y_via_builtin{{.*}}(
uintptr_t get_offset_of_y_via_builtin() {
  // CHECK: [[ENTRY:.*]]:
  // CHECK-NEXT:   ret i64 4
  // CHECK-NEXT: }
  return __builtin_offsetof(struct S, y);
}

#ifdef __cplusplus
}
#endif

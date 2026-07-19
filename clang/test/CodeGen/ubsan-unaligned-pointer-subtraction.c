// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c -w -emit-llvm -o - %s \
// RUN:     -fsanitize=unaligned-pointer-subtraction | \
// RUN:     FileCheck %s --check-prefixes=CHECK,CONLY
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ -w -emit-llvm -o - %s \
// RUN:     -fsanitize=unaligned-pointer-subtraction | \
// RUN:     FileCheck %s
//
// Verify that -fsanitize=unaligned-pointer-subtraction instruments the
// subtraction of two pointers: it checks at runtime that the byte distance is
// an exact multiple of the element size, and skips the check when the element
// size is one (where the remainder is trivially zero).

#ifdef __cplusplus
extern "C" {
#endif

typedef struct { int x, y; } A; // sizeof(A) == 8

// Constant element size: the byte distance is divided by 8, so a check is
// emitted before the exact division.
// CHECK-LABEL: define{{.*}} i64 @f_const(
long f_const(int *p) {
  // CHECK:      %sub.ptr.sub = sub i64 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast
  // CHECK-NEXT: %sub.ptr.rem = srem i64 %sub.ptr.sub, 8, !nosanitize
  // CHECK-NEXT: %sub.ptr.exact = icmp eq i64 %sub.ptr.rem, 0, !nosanitize
  // CHECK-NEXT: br i1 %sub.ptr.exact, label %cont, label %handler.unaligned_pointer_subtraction{{.*}}, !nosanitize
  // CHECK:      call void @__ubsan_handle_unaligned_pointer_subtraction{{.*}}(ptr @{{[0-9]+}}, i64 %sub.ptr.sub, i64 8)
  // CHECK:      %sub.ptr.div = sdiv exact i64 %sub.ptr.sub, 8
  return (A *)(p + 1) - (A *)p;
}

// Element size one (char): no divide is emitted, so no check either.
// CHECK-LABEL: define{{.*}} i64 @f_char(
long f_char(char *a, char *b) {
  // CHECK-NOT: srem
  // CHECK-NOT: __ubsan_handle_unaligned_pointer_subtraction
  // CHECK:     ret i64
  return a - b;
}

#ifdef __cplusplus
} // extern "C"
#endif

// The remaining cases use pointer arithmetic that is only valid in C: VLA
// element types and the GNU void*/function-pointer extensions. They are
// compiled and checked only in the C RUN line (CONLY).
#ifndef __cplusplus

// VLA element type: the divisor is a runtime value (sizeof(int) * n), so the
// remainder check runs against a non-constant divisor.
// CONLY-LABEL: define{{.*}} i64 @f_vla(
long f_vla(int n, int (*p)[n]) {
  int (*q)[n] = (int (*)[n])((char *)p + 4);
  // CONLY:      %sub.ptr.sub = sub i64 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast
  // CONLY-NEXT: %[[ELT:[0-9]+]] = mul nuw i64 4, %{{[0-9]+}}
  // CONLY-NEXT: %sub.ptr.rem = srem i64 %sub.ptr.sub, %[[ELT]], !nosanitize
  // CONLY-NEXT: %sub.ptr.exact = icmp eq i64 %sub.ptr.rem, 0, !nosanitize
  // CONLY:      call void @__ubsan_handle_unaligned_pointer_subtraction{{.*}}(ptr @{{[0-9]+}}, i64 %sub.ptr.sub, i64 %[[ELT]])
  // CONLY:      %sub.ptr.div = sdiv exact i64 %sub.ptr.sub, %[[ELT]]
  return q - p;
}

// void* arithmetic (GNU extension) has element size one: no check.
// CONLY-LABEL: define{{.*}} i64 @f_void(
long f_void(void *a, void *b) {
  // CONLY-NOT: srem
  // CONLY-NOT: __ubsan_handle_unaligned_pointer_subtraction
  // CONLY:     ret i64
  return a - b;
}

typedef void (*fp)(void);

// Function-pointer arithmetic (GNU extension) has element size one: no check.
// CONLY-LABEL: define{{.*}} i64 @f_func(
long f_func(fp a, fp b) {
  // CONLY-NOT: srem
  // CONLY-NOT: __ubsan_handle_unaligned_pointer_subtraction
  // CONLY:     ret i64
  return a - b;
}

#endif // !__cplusplus

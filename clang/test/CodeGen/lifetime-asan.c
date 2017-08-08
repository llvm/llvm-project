// RUN: %clang -target x86_64-linux-gnu -S -emit-llvm -o - -O0 %s | FileCheck %s -check-prefix=CHECK-O0
// RUN: %clang -target x86_64-linux-gnu -S -emit-llvm -o - -O0 \
// RUN:     -fsanitize=address -fsanitize-address-use-after-scope %s | \
// RUN:     FileCheck %s -check-prefix=CHECK-ASAN-USE-AFTER-SCOPE

extern int bar(char *A, int n);

// CHECK-O0-NOT: @llvm.lifetime.start
int foo(int n) {
  if (n) {
    // CHECK-ASAN-USE-AFTER-SCOPE: @llvm.lifetime.start.p0i8(i64 10, i8* {{.*}})
    char A[10];
    return bar(A, 1);
    // CHECK-ASAN-USE-AFTER-SCOPE: @llvm.lifetime.end.p0i8(i64 10, i8* {{.*}})
  } else {
    // CHECK-ASAN-USE-AFTER-SCOPE: @llvm.lifetime.start.p0i8(i64 20, i8* {{.*}})
    char A[20];
    return bar(A, 2);
    // CHECK-ASAN-USE-AFTER-SCOPE: @llvm.lifetime.end.p0i8(i64 20, i8* {{.*}})
  }
}

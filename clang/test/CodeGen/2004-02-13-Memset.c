// RUN: %clang_cc1  %s -emit-llvm -o - | FileCheck %s

typedef __SIZE_TYPE__ size_t;
void *memset(void*, int, size_t);
void bzero(void*, size_t);

void test(int* X, char *Y) {
  // CHECK: call void @llvm.memset{{.*}}i8 4, {{.*}} 1000
  memset(X, 4, 1000);
  // CHECK: call void @llvm.memset{{.*}}i8 0, {{.*}} 100
  bzero(Y, 100);
}

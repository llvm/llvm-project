// RUN: %clang_cc1  %s -emit-llvm -o - | grep llvm.memset | count 3

typedef __SIZE_TYPE__ size_t;
void *memset(void*, int, size_t);
void bzero(void*, size_t);

void test(int* X, char *Y) {
  // CHECK: call i8* llvm.memset
  memset(X, 4, 1000);
  // CHECK: call void bzero
  bzero(Y, 100);
}

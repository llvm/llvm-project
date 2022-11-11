// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm < %s| FileCheck %s

typedef __SIZE_TYPE__ size_t;

void *mempcpy(void *, void const *, size_t);

char *test(char *d, char *s, size_t n) {
  // CHECK:      call void @llvm.memcpy.p0.p0.i64(ptr {{.*}} %[[REG1:[^ ]+]], ptr {{.*}} %1, i64 %[[REG2:[^ ]+]], i1 false)
  // CHECK-NEXT: %[[REGr:[^ ]+]] = getelementptr inbounds i8, ptr %[[REG1]], i64 %[[REG2]]
  // CHECK-NEXT: ret ptr %[[REGr]]
  return mempcpy(d, s, n);
}

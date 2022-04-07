// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-unknown-linux-gnu -emit-llvm < %s| FileCheck %s

typedef __SIZE_TYPE__ size_t;

void *mempcpy(void *, void const *, size_t);

char *test(char *d, char *s, size_t n) {
  // CHECK:      call void @llvm.memcpy.p0i8.p0i8.i64(i8* {{.*}} %[[REG1:[^ ]+]], i8* {{.*}} %1, i64 %[[REG2:[^ ]+]], i1 false)
  // CHECK-NEXT: %[[REGr:[^ ]+]] = getelementptr inbounds i8, i8* %[[REG1]], i64 %[[REG2]]
  // CHECK-NEXT: ret i8* %[[REGr]]
  return mempcpy(d, s, n);
}

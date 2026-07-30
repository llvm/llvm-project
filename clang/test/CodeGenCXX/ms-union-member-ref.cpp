// RUN: %clang_cc1 -triple=i686-pc-win32 -fms-extensions %s -emit-llvm -o- | FileCheck %s

union A {
  int *&ref;
  int **ptr;
};

int *f1(A *a) {
  return a->ref;
}
// CHECK-LABEL: define {{.*}}ptr @"?f1@@YAPAHPATA@@@Z"(ptr noundef %a)
// CHECK:       [[A_ADDR:%[^[:space:]]+]] = load ptr, ptr %{{.*}}
// CHECK:       [[IPP:%[^[:space:]]+]] = load ptr, ptr [[A_ADDR]]
// CHECK:       [[IP:%[^[:space:]]+]]  = load ptr, ptr [[IPP]]
// CHECK:       ret ptr [[IP]]

void f2(A *a) {
  *a->ref = 1;
}
// CHECK-LABEL: define {{.*}}void @"?f2@@YAXPATA@@@Z"(ptr noundef %a)
// CHECK:       [[A_ADDR:%[^[:space:]]+]] = load ptr, ptr %{{.*}}
// CHECK:       [[IPP:%[^[:space:]]+]] = load ptr, ptr [[A_ADDR]]
// CHECK:       [[IP:%[^[:space:]]+]]  = load ptr, ptr [[IPP]]
// CHECK:       store i32 1, ptr [[IP]]

bool f3(A *a, int *b) {
  return a->ref != b;
}
// CHECK-LABEL: define {{.*}}i1 @"?f3@@YA_NPATA@@PAH@Z"(ptr noundef %a, ptr noundef %b)
// CHECK:       [[A_ADDR:%[^[:space:]]+]] = load ptr, ptr %{{.*}}
// CHECK:       [[IPP:%[^[:space:]]+]] = load ptr, ptr [[A_ADDR]]
// CHECK:       [[IP:%[^[:space:]]+]]  = load ptr, ptr [[IPP]]
// CHECK:       [[IP2:%[^[:space:]]+]]  = load ptr, ptr %b.addr
// CHECK:       icmp ne ptr [[IP]], [[IP2]]

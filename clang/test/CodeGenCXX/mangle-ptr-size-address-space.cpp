// RUN: %clang_cc1 -fms-extensions -emit-llvm -triple x86_64-linux-gnu -o - %s | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -fms-extensions -emit-llvm -triple x86_64-windows-msvc -o - %s | FileCheck %s --check-prefixes=WIN

// CHECK-LABEL: define {{.*}}void @_Z2f0PU10ptr32_sptri
// WIN-LABEL: define {{.*}}void @"?f0@@YAXPAH@Z"
void f0(int * __ptr32 p) {}

// CHECK-LABEL: define {{.*}}ptr addrspace(271) @_Z2f1PU10ptr32_sptri
// WIN-LABEL: define {{.*}}ptr addrspace(271) @"?f1@@YAPAXPAH@Z"
void * __ptr32 __uptr f1(int * __ptr32 p) { return 0; }

// CHECK-LABEL: define {{.*}}void @_Z2f2Pi
// WIN-LABEL: define {{.*}}void @"?f2@@YAXPEAH@Z"
void f2(int * __ptr64 p) {}

// CHECK-LABEL: define {{.*}}ptr @_Z2f3Pi
// WIN-LABEL: define {{.*}}ptr @"?f3@@YAPEAXPEAH@Z"
void * __ptr64 f3(int * __ptr64 p) { return 0; }

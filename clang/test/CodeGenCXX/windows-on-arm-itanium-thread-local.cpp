// RUN: %clang_cc1 -triple thumbv7--windows-itanium -fdeclspec -fms-compatibility -fms-compatibility-version=19.0 -emit-llvm -o - %s | FileCheck %s

void *g();
thread_local static void *c = g();
void f(void *p) {
  c = p;
}

// CHECK-LABEL: @_Z1fPv(ptr noundef %p)
// CHECK-NOT: call ptr @_ZTWL1c()
// CHECK: call arm_aapcs_vfpcc ptr @_ZTWL1c()


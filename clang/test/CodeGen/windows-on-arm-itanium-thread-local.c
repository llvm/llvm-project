// RUN: %clang_cc1 -triple thumbv7--windows-itanium -fdeclspec -fms-compatibility -fms-compatibility-version=19.0 -S -emit-llvm -o - %s | FileCheck %s

__declspec(thread) static void *c;
void f(void *p) {
  c = p;
}

// CHECK-LABEL: @f(i8* %p)
// CHECK-NOT: call i8** @_ZTWL1c()
// CHECK: call arm_aapcs_vfpcc i8** @_ZTWL1c()


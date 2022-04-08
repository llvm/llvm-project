// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin10.0.0 -emit-llvm -o - %s -std=c++11 -debug-info-kind=limited | FileCheck %s

void crash() {
  volatile char *ptr = 0;
  char x = *ptr;
}

int test() {
  crash();
  return 1;
}

static int i = test();
__attribute__((nodebug)) static int j = test();
static int k = test();

int main(void) {}

// CHECK-LABEL: define internal void @__cxx_global_var_init()
// CHECK-NOT: __cxx_global_var_init
// CHECK: %[[C0:.+]] = call noundef i32 @_Z4testv(), !dbg ![[LINE:.*]]
// CHECK-NOT: __cxx_global_var_init
// CHECK: store i32 %[[C0]], i32* @_ZL1i, align 4, !dbg
// 
// CHECK-LABEL: define internal void @__cxx_global_var_init.1()
// CHECK-NOT: dbg
// CHECK: %[[C1:.+]] = call noundef i32 @_Z4testv()
// CHECK-NOT: dbg
// CHECK: store i32 %[[C1]], i32* @_ZL1j, align 4
//
// CHECK-LABEL: define internal void @__cxx_global_var_init.2()
// CHECK-NOT: __cxx_global_var_init
// CHECK: %[[C2:.+]] = call noundef i32 @_Z4testv(), !dbg ![[LINE2:.*]]
// CHECK-NOT: __cxx_global_var_init
// CHECK: store i32 %[[C2]], i32* @_ZL1k, align 4, !dbg
// 
// CHECK: ![[LINE]] = !DILocation(line: 13,
// CHECK: ![[LINE2]] = !DILocation(line: 15,

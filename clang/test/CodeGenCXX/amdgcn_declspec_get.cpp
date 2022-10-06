// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm -O3 -fdeclspec \
// RUN:     -disable-llvm-passes -o - %s | FileCheck %s

int get_x();

struct A {
   __declspec(property(get = _get_x)) int x;
   static int _get_x(void) {
     return get_x();
   };
};

extern const A a;

// CHECK-LABEL: define{{.*}} void @_Z4testv()
// CHECK:  %i = alloca i32, align 4, addrspace(5)
// CHECK:  %[[ii:.*]] = addrspacecast ptr addrspace(5) %i to ptr
// CHECK:  call void @llvm.lifetime.start.p5(i64 4, ptr addrspace(5) %i)
// CHECK:  %call = call noundef i32 @_ZN1A6_get_xEv()
// CHECK:  store i32 %call, ptr %[[ii]]
// CHECK:  call void @llvm.lifetime.end.p5(i64 4, ptr addrspace(5) %i)
void test()
{
  int i = a.x;
}

// RUN: %clang_cc1 -no-opaque-pointers -triple amdgcn-amd-amdhsa -emit-llvm -O3 -fdeclspec \
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
// CHECK:  %[[ii:.*]] = addrspacecast i32 addrspace(5)* %i to i32*
// CHECK:  %[[cast:.*]] = bitcast i32 addrspace(5)* %i to i8 addrspace(5)*
// CHECK:  call void @llvm.lifetime.start.p5i8(i64 4, i8 addrspace(5)* %[[cast]])
// CHECK:  %call = call noundef i32 @_ZN1A6_get_xEv()
// CHECK:  store i32 %call, i32* %[[ii]]
// CHECK:  %[[cast2:.*]] = bitcast i32 addrspace(5)* %i to i8 addrspace(5)*
// CHECK:  call void @llvm.lifetime.end.p5i8(i64 4, i8 addrspace(5)* %[[cast2]])
void test()
{
  int i = a.x;
}

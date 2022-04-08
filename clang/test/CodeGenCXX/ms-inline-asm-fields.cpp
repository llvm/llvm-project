// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -no-opaque-pointers %s -triple i386-apple-darwin10 -fasm-blocks -emit-llvm -o - | FileCheck %s

struct A {
  int a1;
  int a2;
  struct B {
    int b1;
    int b2;
  } a3;
};

namespace asdf {
A a_global;
}

extern "C" int test_param_field(A p) {
// CHECK: define{{.*}} i32 @test_param_field(%struct.A* noundef byval(%struct.A) align 4 %p)
// CHECK: getelementptr inbounds %struct.A, %struct.A* %p, i32 0, i32 0
// CHECK: call i32 asm sideeffect inteldialect "mov eax, $1"
// CHECK: ret i32
  __asm mov eax, p.a1
}

extern "C" int test_namespace_global() {
// CHECK: define{{.*}} i32 @test_namespace_global()
// CHECK: call i32 asm sideeffect inteldialect "mov eax, $1", "{{.*}}"(i32* elementtype(i32) getelementptr inbounds (%struct.A, %struct.A* @_ZN4asdf8a_globalE, i32 0, i32 2, i32 1))
// CHECK: ret i32
  __asm mov eax, asdf::a_global.a3.b2
}

template <bool Signed>
struct make_storage_type {
  struct type {
    struct B {
      int a;
      int x;
    } b;
  };
};

template <bool Signed>
struct msvc_dcas_x86 {
  typedef typename make_storage_type<Signed>::type storage_type;
  void store() __asm("PR26001") {
    storage_type p;
    __asm mov edx, p.b.x;
  }
};

template void msvc_dcas_x86<false>::store();
// CHECK: define weak_odr void @"\01PR26001"(
// CHECK: %[[P:.*]] = alloca %"struct.make_storage_type<false>::type", align 4
// CHECK: %[[B:.*]] = getelementptr inbounds %"struct.make_storage_type<false>::type", %"struct.make_storage_type<false>::type"* %[[P]], i32 0, i32 0
// CHECK: %[[X:.*]] = getelementptr inbounds %"struct.make_storage_type<false>::type::B", %"struct.make_storage_type<false>::type::B"* %[[B]], i32 0, i32 1
// CHECK: call void asm sideeffect inteldialect "mov edx, dword ptr $0", "*m,~{edx},~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) %[[X]])

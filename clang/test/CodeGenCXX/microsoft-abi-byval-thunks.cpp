// RUN: %clang_cc1 %s -fno-rtti -triple=i686-pc-win32 -emit-llvm -o - | FileCheck --check-prefix=CHECK32 %s
// RUN: %clang_cc1 %s -fno-rtti -triple=x86_64-pc-win32 -emit-llvm -o - | FileCheck --check-prefix=CHECK64 %s

namespace byval_thunk {
struct Agg {
  Agg();
  Agg(const Agg &);
  ~Agg();
  int x;
};

struct A { virtual void foo(Agg x); };
struct B { virtual void foo(Agg x); };
struct C : A, B { C(); virtual void foo(Agg x); };
C::C() {} // force emission

// CHECK32-LABEL: define linkonce_odr dso_local x86_thiscallcc void @"?foo@C@byval_thunk@@W3AEXUAgg@2@@Z"
// CHECK32:             (ptr noundef %this, ptr inalloca(<{ %"struct.byval_thunk::Agg" }>) %0)
// CHECK32:   getelementptr i8, ptr %{{.*}}, i32 -4
// CHECK32:   musttail call x86_thiscallcc void @"?foo@C@byval_thunk@@UAEXUAgg@2@@Z"
// CHECK32:       (ptr noundef %{{.*}}, ptr inalloca(<{ %"struct.byval_thunk::Agg" }>) %0)
// CHECK32-NEXT: ret void

// CHECK64-LABEL: define linkonce_odr dso_local void @"?foo@C@byval_thunk@@W7EAAXUAgg@2@@Z"
// CHECK64:             (ptr noundef %this, ptr noundef %x)
// CHECK64:   getelementptr i8, ptr %{{.*}}, i32 -8
// CHECK64:   call void @"?foo@C@byval_thunk@@UEAAXUAgg@2@@Z"
// CHECK64:       (ptr {{[^,]*}} %{{.*}}, ptr noundef %x)
// CHECK64-NOT: call
// CHECK64:   ret void
}

namespace stdcall_thunk {
struct Agg {
  Agg();
  Agg(const Agg &);
  ~Agg();
  int x;
};

struct A { virtual void __stdcall foo(Agg x); };
struct B { virtual void __stdcall foo(Agg x); };
struct C : A, B { C(); virtual void __stdcall foo(Agg x); };
C::C() {} // force emission

// CHECK32-LABEL: define linkonce_odr dso_local x86_stdcallcc void @"?foo@C@stdcall_thunk@@W3AGXUAgg@2@@Z"
// CHECK32:             (ptr inalloca(<{ ptr, %"struct.stdcall_thunk::Agg" }>) %0)
// CHECK32:   %[[this_slot:[^ ]*]] = getelementptr inbounds <{ ptr, %"struct.stdcall_thunk::Agg" }>, ptr %0, i32 0, i32 0
// CHECK32:   load ptr, ptr %[[this_slot]]
// CHECK32:   getelementptr i8, ptr %{{.*}}, i32 -4
// CHECK32:   store ptr %{{.*}}, ptr %[[this_slot]]
// CHECK32:   musttail call x86_stdcallcc void @"?foo@C@stdcall_thunk@@UAGXUAgg@2@@Z"
// CHECK32:       (ptr  inalloca(<{ ptr, %"struct.stdcall_thunk::Agg" }>) %0)
// CHECK32-NEXT: ret void

// CHECK64-LABEL: define linkonce_odr dso_local void @"?foo@C@stdcall_thunk@@W7EAAXUAgg@2@@Z"
// CHECK64:             (ptr noundef %this, ptr noundef %x)
// CHECK64:   getelementptr i8, ptr %{{.*}}, i32 -8
// CHECK64:   call void @"?foo@C@stdcall_thunk@@UEAAXUAgg@2@@Z"
// CHECK64:       (ptr {{[^,]*}} %{{.*}}, ptr noundef %x)
// CHECK64-NOT: call
// CHECK64:   ret void
}

namespace sret_thunk {
struct Agg {
  Agg();
  Agg(const Agg &);
  ~Agg();
  int x;
};

struct A { virtual Agg __cdecl foo(Agg x); };
struct B { virtual Agg __cdecl foo(Agg x); };
struct C : A, B { C(); virtual Agg __cdecl foo(Agg x); };
C::C() {} // force emission

// CHECK32-LABEL: define linkonce_odr dso_local ptr @"?foo@C@sret_thunk@@W3AA?AUAgg@2@U32@@Z"
// CHECK32:             (ptr inalloca(<{ ptr, ptr, %"struct.sret_thunk::Agg" }>) %0)
// CHECK32:   %[[this_slot:[^ ]*]] = getelementptr inbounds <{ ptr, ptr, %"struct.sret_thunk::Agg" }>, ptr %0, i32 0, i32 0
// CHECK32:   load ptr, ptr %[[this_slot]]
// CHECK32:   getelementptr i8, ptr %{{.*}}, i32 -4
// CHECK32:   store ptr %{{.*}}, ptr %[[this_slot]]
// CHECK32:   %[[rv:[^ ]*]] = musttail call ptr @"?foo@C@sret_thunk@@UAA?AUAgg@2@U32@@Z"
// CHECK32:       (ptr  inalloca(<{ ptr, ptr, %"struct.sret_thunk::Agg" }>) %0)
// CHECK32-NEXT: ret ptr %[[rv]]

// CHECK64-LABEL: define linkonce_odr dso_local void @"?foo@C@sret_thunk@@W7EAA?AUAgg@2@U32@@Z"
// CHECK64:             (ptr noundef %this, ptr noalias sret(%"struct.sret_thunk::Agg") align 4 %agg.result, ptr noundef %x)
// CHECK64:   getelementptr i8, ptr %{{.*}}, i32 -8
// CHECK64:   call void @"?foo@C@sret_thunk@@UEAA?AUAgg@2@U32@@Z"
// CHECK64:       (ptr {{[^,]*}} %{{.*}}, ptr sret(%"struct.sret_thunk::Agg") align 4 %agg.result, ptr noundef %x)
// CHECK64-NOT: call
// CHECK64:   ret void
}

#if 0
// FIXME: When we extend LLVM IR to allow forwarding of varargs through musttail
// calls, use this test.
namespace variadic_thunk {
struct Agg {
  Agg();
  Agg(const Agg &);
  ~Agg();
  int x;
};

struct A { virtual void foo(Agg x, ...); };
struct B { virtual void foo(Agg x, ...); };
struct C : A, B { C(); virtual void foo(Agg x, ...); };
C::C() {} // force emission
}
#endif

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
// CHECK32:             (%"struct.byval_thunk::C"* %this, <{ %"struct.byval_thunk::Agg" }>* inalloca %0)
// CHECK32:   getelementptr i8, i8* %{{.*}}, i32 -4
// CHECK32:   musttail call x86_thiscallcc void @"?foo@C@byval_thunk@@UAEXUAgg@2@@Z"
// CHECK32:       (%"struct.byval_thunk::C"* %{{.*}}, <{ %"struct.byval_thunk::Agg" }>* inalloca %0)
// CHECK32-NEXT: ret void

// CHECK64-LABEL: define linkonce_odr dso_local void @"?foo@C@byval_thunk@@W7EAAXUAgg@2@@Z"
// CHECK64:             (%"struct.byval_thunk::C"* {{[^,]*}} %this, %"struct.byval_thunk::Agg"* %x)
// CHECK64:   getelementptr i8, i8* %{{.*}}, i32 -8
// CHECK64:   call void @"?foo@C@byval_thunk@@UEAAXUAgg@2@@Z"
// CHECK64:       (%"struct.byval_thunk::C"* {{[^,]*}} %{{.*}}, %"struct.byval_thunk::Agg"* %x)
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
// CHECK32:             (<{ %"struct.stdcall_thunk::C"*, %"struct.stdcall_thunk::Agg" }>* inalloca %0)
// CHECK32:   %[[this_slot:[^ ]*]] = getelementptr inbounds <{ %"struct.stdcall_thunk::C"*, %"struct.stdcall_thunk::Agg" }>, <{ %"struct.stdcall_thunk::C"*, %"struct.stdcall_thunk::Agg" }>* %0, i32 0, i32 0
// CHECK32:   load %"struct.stdcall_thunk::C"*, %"struct.stdcall_thunk::C"** %[[this_slot]]
// CHECK32:   getelementptr i8, i8* %{{.*}}, i32 -4
// CHECK32:   store %"struct.stdcall_thunk::C"* %{{.*}}, %"struct.stdcall_thunk::C"** %[[this_slot]]
// CHECK32:   musttail call x86_stdcallcc void @"?foo@C@stdcall_thunk@@UAGXUAgg@2@@Z"
// CHECK32:       (<{ %"struct.stdcall_thunk::C"*, %"struct.stdcall_thunk::Agg" }>*  inalloca %0)
// CHECK32-NEXT: ret void

// CHECK64-LABEL: define linkonce_odr dso_local void @"?foo@C@stdcall_thunk@@W7EAAXUAgg@2@@Z"
// CHECK64:             (%"struct.stdcall_thunk::C"* {{[^,]*}} %this, %"struct.stdcall_thunk::Agg"* %x)
// CHECK64:   getelementptr i8, i8* %{{.*}}, i32 -8
// CHECK64:   call void @"?foo@C@stdcall_thunk@@UEAAXUAgg@2@@Z"
// CHECK64:       (%"struct.stdcall_thunk::C"* {{[^,]*}} %{{.*}}, %"struct.stdcall_thunk::Agg"* %x)
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

// CHECK32-LABEL: define linkonce_odr dso_local %"struct.sret_thunk::Agg"* @"?foo@C@sret_thunk@@W3AA?AUAgg@2@U32@@Z"
// CHECK32:             (<{ %"struct.sret_thunk::C"*, %"struct.sret_thunk::Agg"*, %"struct.sret_thunk::Agg" }>* inalloca %0)
// CHECK32:   %[[this_slot:[^ ]*]] = getelementptr inbounds <{ %"struct.sret_thunk::C"*, %"struct.sret_thunk::Agg"*, %"struct.sret_thunk::Agg" }>, <{ %"struct.sret_thunk::C"*, %"struct.sret_thunk::Agg"*, %"struct.sret_thunk::Agg" }>* %0, i32 0, i32 0
// CHECK32:   load %"struct.sret_thunk::C"*, %"struct.sret_thunk::C"** %[[this_slot]]
// CHECK32:   getelementptr i8, i8* %{{.*}}, i32 -4
// CHECK32:   store %"struct.sret_thunk::C"* %{{.*}}, %"struct.sret_thunk::C"** %[[this_slot]]
// CHECK32:   %[[rv:[^ ]*]] = musttail call %"struct.sret_thunk::Agg"* @"?foo@C@sret_thunk@@UAA?AUAgg@2@U32@@Z"
// CHECK32:       (<{ %"struct.sret_thunk::C"*, %"struct.sret_thunk::Agg"*, %"struct.sret_thunk::Agg" }>*  inalloca %0)
// CHECK32-NEXT: ret %"struct.sret_thunk::Agg"* %[[rv]]

// CHECK64-LABEL: define linkonce_odr dso_local void @"?foo@C@sret_thunk@@W7EAA?AUAgg@2@U32@@Z"
// CHECK64:             (%"struct.sret_thunk::C"* {{[^,]*}} %this, %"struct.sret_thunk::Agg"* noalias sret(%"struct.sret_thunk::Agg") align 4 %agg.result, %"struct.sret_thunk::Agg"* %x)
// CHECK64:   getelementptr i8, i8* %{{.*}}, i32 -8
// CHECK64:   call void @"?foo@C@sret_thunk@@UEAA?AUAgg@2@U32@@Z"
// CHECK64:       (%"struct.sret_thunk::C"* {{[^,]*}} %{{.*}}, %"struct.sret_thunk::Agg"* sret(%"struct.sret_thunk::Agg") align 4 %agg.result, %"struct.sret_thunk::Agg"* %x)
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

// RUN: %clang_cc1 -fno-rtti -emit-llvm -triple=i386-pc-win32 %s -o - | FileCheck %s

// In each test case, we have two member pointers whose thunks have the same
// vtable offset and same mangling, but their prototypes conflict.  The
// arguments and return type may differ.  Therefore, we have to bitcast the
// function prototype.  Unfortunately, if the return types differ, LLVM's
// optimizers can get upset.

namespace num_params {
struct A { virtual void a(int); };
struct B { virtual void b(int, int); };
struct C : A, B {
  virtual void a(int);
  virtual void b(int, int);
};
void f(C *c) {
  (c->*(&C::a))(0);
  (c->*(&C::b))(0, 0);
}
}

// CHECK-LABEL: define void @"\01?f@num_params@@YAXPAUC@1@@Z"(%"struct.num_params::C"* %c)
// CHECK: call x86_thiscallcc void bitcast (void (%"struct.num_params::C"*, ...)* @"\01??_9C@num_params@@$BA@AE" to void (%"struct.num_params::C"*, i32)*)(%"struct.num_params::C"* %{{.*}}, i32 0)
// CHECK: call x86_thiscallcc void bitcast (void (%"struct.num_params::C"*, ...)* @"\01??_9C@num_params@@$BA@AE" to void (%"struct.num_params::C"*, i32, i32)*)(%"struct.num_params::C"* %{{.*}}, i32 0, i32 0)

// CHECK-LABEL: define linkonce_odr x86_thiscallcc void @"\01??_9C@num_params@@$BA@AE"(%"struct.num_params::C"* %this, ...) {{.*}} comdat
// CHECK: musttail call x86_thiscallcc void (%"struct.num_params::C"*, ...) %{{.*}}(%"struct.num_params::C"* %{{.*}}, ...)
// CHECK-NEXT: ret void

namespace i64_return {
struct A { virtual int a(); };
struct B { virtual long long b(); };
struct C : A, B {
  virtual int a();
  virtual long long b();
};
long long f(C *c) {
  int x = (c->*(&C::a))();
  long long y = (c->*(&C::b))();
  return x + y;
}
}

// CHECK-LABEL: define i64 @"\01?f@i64_return@@YA_JPAUC@1@@Z"(%"struct.i64_return::C"* %c)
// CHECK: call x86_thiscallcc i32 bitcast (void (%"struct.i64_return::C"*, ...)* @"\01??_9C@i64_return@@$BA@AE" to i32 (%"struct.i64_return::C"*)*)(%"struct.i64_return::C"* %{{.*}})
// CHECK: call x86_thiscallcc i64 bitcast (void (%"struct.i64_return::C"*, ...)* @"\01??_9C@i64_return@@$BA@AE" to i64 (%"struct.i64_return::C"*)*)(%"struct.i64_return::C"* %{{.*}})

// CHECK-LABEL: define linkonce_odr x86_thiscallcc void @"\01??_9C@i64_return@@$BA@AE"(%"struct.i64_return::C"* %this, ...) {{.*}} comdat
// CHECK: musttail call x86_thiscallcc void (%"struct.i64_return::C"*, ...) %{{.*}}(%"struct.i64_return::C"* %{{.*}}, ...)
// CHECK-NEXT: ret void

namespace sret {
struct Big { int big[32]; };
struct A { virtual int a(); };
struct B { virtual Big b(); };
struct C : A, B {
  virtual int a();
  virtual Big b();
};
void f(C *c) {
  (c->*(&C::a))();
  Big b((c->*(&C::b))());
}
}

// CHECK-LABEL: define void @"\01?f@sret@@YAXPAUC@1@@Z"(%"struct.sret::C"* %c)
// CHECK: call x86_thiscallcc i32 bitcast (void (%"struct.sret::C"*, ...)* @"\01??_9C@sret@@$BA@AE" to i32 (%"struct.sret::C"*)*)(%"struct.sret::C"* %{{.*}})
// CHECK: call x86_thiscallcc void bitcast (void (%"struct.sret::C"*, ...)* @"\01??_9C@sret@@$BA@AE" to void (%"struct.sret::C"*, %"struct.sret::Big"*)*)(%"struct.sret::C"* %{{.*}}, %"struct.sret::Big"* sret %{{.*}})

// CHECK-LABEL: define linkonce_odr x86_thiscallcc void @"\01??_9C@sret@@$BA@AE"(%"struct.sret::C"* %this, ...) {{.*}} comdat
// CHECK: musttail call x86_thiscallcc void (%"struct.sret::C"*, ...) %{{.*}}(%"struct.sret::C"* %{{.*}}, ...)
// CHECK-NEXT: ret void

namespace cdecl_inalloca {
// Fairly evil, since now we end up doing an inalloca-style call through a
// thunk that doesn't use inalloca.  Hopefully the stacks line up?
struct Big {
  Big();
  ~Big();
  int big[32];
};
struct A { virtual void __cdecl a(); };
struct B { virtual void __cdecl b(Big); };
struct C : A, B {
  virtual void __cdecl a();
  virtual void __cdecl b(Big);
};
void f(C *c) {
  Big b;
  (c->*(&C::a))();
  ((c->*(&C::b))(b));
}
}

// CHECK-LABEL: define void @"\01?f@cdecl_inalloca@@YAXPAUC@1@@Z"(%"struct.cdecl_inalloca::C"* %c)
// CHECK: call void bitcast (void (%"struct.cdecl_inalloca::C"*, ...)* @"\01??_9C@cdecl_inalloca@@$BA@AA" to void (%"struct.cdecl_inalloca::C"*)*)(%"struct.cdecl_inalloca::C"* %{{.*}})
// CHECK: call void bitcast (void (%"struct.cdecl_inalloca::C"*, ...)* @"\01??_9C@cdecl_inalloca@@$BA@AA" to void (<{ %"struct.cdecl_inalloca::C"*, %"struct.cdecl_inalloca::Big" }>*)*)(<{ %"struct.cdecl_inalloca::C"*, %"struct.cdecl_inalloca::Big" }>* inalloca %{{.*}})

// CHECK-LABEL: define linkonce_odr void @"\01??_9C@cdecl_inalloca@@$BA@AA"(%"struct.cdecl_inalloca::C"* %this, ...) {{.*}} comdat
// CHECK: musttail call void (%"struct.cdecl_inalloca::C"*, ...) %{{.*}}(%"struct.cdecl_inalloca::C"* %{{.*}}, ...)
// CHECK-NEXT: ret void

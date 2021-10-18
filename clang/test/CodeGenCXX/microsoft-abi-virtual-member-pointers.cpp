// RUN: %clang_cc1 -std=c++11 -fno-rtti -emit-llvm -triple=i386-pc-win32 %s -o - | FileCheck %s --check-prefix=CHECK32
// RUN: %clang_cc1 -std=c++11 -fno-rtti -emit-llvm -triple=x86_64-pc-win32 %s -o - | FileCheck %s --check-prefix=CHECK64

struct S {
  int x, y, z;
};

// U is not trivially copyable, and requires inalloca to pass by value.
struct U {
  int u;
  U();
  ~U();
  U(const U &);
};

struct B;

struct C {
  virtual void foo();
  virtual int bar(int, double);
  virtual S baz(int);
  virtual S qux(U);
  virtual void thud(...);
  virtual void (B::*plugh())();
};

namespace {
struct D {
  virtual void foo();
};
}

void f() {
  void (C::*ptr)();
  ptr = &C::foo;
  ptr = &C::foo; // Don't crash trying to define the thunk twice :)

  int (C::*ptr2)(int, double);
  ptr2 = &C::bar;

  S (C::*ptr3)(int);
  ptr3 = &C::baz;

  void (D::*ptr4)();
  ptr4 = &D::foo;

  S (C::*ptr5)(U);
  ptr5 = &C::qux;

  void (C::*ptr6)(...);
  ptr6 = &C::thud;

  auto ptr7 = &C::plugh;


// CHECK32-LABEL: define dso_local void @"?f@@YAXXZ"()
// CHECK32: store i8* bitcast (void (%struct.C*, ...)* @"??_9C@@$BA@AE" to i8*), i8** %ptr
// CHECK32: store i8* bitcast (void (%struct.C*, ...)* @"??_9C@@$B3AE" to i8*), i8** %ptr2
// CHECK32: store i8* bitcast (void (%struct.C*, ...)* @"??_9C@@$B7AE" to i8*), i8** %ptr3
// CHECK32: store i8* bitcast (void (%"struct.(anonymous namespace)::D"*, ...)* @"??_9D@?A0x{{[^@]*}}@@$BA@AE" to i8*), i8** %ptr4
// CHECK32: }
//
// CHECK64-LABEL: define dso_local void @"?f@@YAXXZ"()
// CHECK64: store i8* bitcast (void (%struct.C*, ...)* @"??_9C@@$BA@AA" to i8*), i8** %ptr
// CHECK64: store i8* bitcast (void (%struct.C*, ...)* @"??_9C@@$B7AA" to i8*), i8** %ptr2
// CHECK64: store i8* bitcast (void (%struct.C*, ...)* @"??_9C@@$BBA@AA" to i8*), i8** %ptr3
// CHECK64: store i8* bitcast (void (%"struct.(anonymous namespace)::D"*, ...)* @"??_9D@?A0x{{[^@]*}}@@$BA@AA" to i8*), i8** %ptr
// CHECK64: }
}


// Thunk for calling the 1st virtual function in C with no parameters.
// CHECK32-LABEL: define linkonce_odr x86_thiscallcc void @"??_9C@@$BA@AE"(%struct.C* %this, ...)
// CHECK32: #[[ATTR:[0-9]+]]
// CHECK32-NOT:             unnamed_addr
// CHECK32:                 comdat
// CHECK32: [[VPTR:%.*]] = getelementptr inbounds void (%struct.C*, ...)*, void (%struct.C*, ...)** %{{.*}}, i64 0
// CHECK32: [[CALLEE:%.*]] = load void (%struct.C*, ...)*, void (%struct.C*, ...)** [[VPTR]]
// CHECK32: musttail call x86_thiscallcc void (%struct.C*, ...) [[CALLEE]](%struct.C* %{{.*}}, ...)
// CHECK32-NEXT: ret void
// CHECK32: }
//
// CHECK64-LABEL: define linkonce_odr void @"??_9C@@$BA@AA"(%struct.C* %this, ...)
// CHECK64: #[[ATTR:[0-9]+]]
// CHECK64-NOT:             unnamed_addr
// CHECK64:                 comdat
// CHECK64: [[VPTR:%.*]] = getelementptr inbounds void (%struct.C*, ...)*, void (%struct.C*, ...)** %{{.*}}, i64 0
// CHECK64: [[CALLEE:%.*]] = load void (%struct.C*, ...)*, void (%struct.C*, ...)** [[VPTR]]
// CHECK64: musttail call void (%struct.C*, ...) [[CALLEE]](%struct.C* %{{.*}}, ...)
// CHECK64-NEXT: ret void
// CHECK64: }

// Thunk for calling the 2nd virtual function in C, taking int and double as parameters, returning int.
// CHECK32-LABEL: define linkonce_odr x86_thiscallcc void @"??_9C@@$B3AE"(%struct.C* %this, ...)
// CHECK32: #[[ATTR]] comdat
// CHECK32: [[VPTR:%.*]] = getelementptr inbounds void (%struct.C*, ...)*, void (%struct.C*, ...)** %{{.*}}, i64 1
// CHECK32: [[CALLEE:%.*]] = load void (%struct.C*, ...)*, void (%struct.C*, ...)** [[VPTR]]
// CHECK32: musttail call x86_thiscallcc void (%struct.C*, ...) [[CALLEE]](%struct.C* %{{.*}}, ...)
// CHECK32-NEXT: ret void
// CHECK32: }
//
// CHECK64-LABEL: define linkonce_odr void @"??_9C@@$B7AA"(%struct.C* %this, ...)
// CHECK64: #[[ATTR]] comdat
// CHECK64: [[VPTR:%.*]] = getelementptr inbounds void (%struct.C*, ...)*, void (%struct.C*, ...)** %{{.*}}, i64 1
// CHECK64: [[CALLEE:%.*]] = load void (%struct.C*, ...)*, void (%struct.C*, ...)** [[VPTR]]
// CHECK64: musttail call void (%struct.C*, ...) [[CALLEE]](%struct.C* %{{.*}}, ...)
// CHECK64-NEXT: ret void
// CHECK64: }

// Thunk for calling the 3rd virtual function in C, taking an int parameter, returning a struct.
// CHECK32-LABEL: define linkonce_odr x86_thiscallcc void @"??_9C@@$B7AE"(%struct.C* %this, ...)
// CHECK32: #[[ATTR]] comdat
// CHECK32: [[VPTR:%.*]] = getelementptr inbounds void (%struct.C*, ...)*, void (%struct.C*, ...)** %{{.*}}, i64 2
// CHECK32: [[CALLEE:%.*]] = load void (%struct.C*, ...)*, void (%struct.C*, ...)** [[VPTR]]
// CHECK32: musttail call x86_thiscallcc void (%struct.C*, ...) [[CALLEE]](%struct.C* %{{.*}}, ...)
// CHECK32-NEXT: ret void
// CHECK32: }
//
// CHECK64-LABEL: define linkonce_odr void @"??_9C@@$BBA@AA"(%struct.C* %this, ...)
// CHECK64: #[[ATTR]] comdat
// CHECK64: [[VPTR:%.*]] = getelementptr inbounds void (%struct.C*, ...)*, void (%struct.C*, ...)** %{{.*}}, i64 2
// CHECK64: [[CALLEE:%.*]] = load void (%struct.C*, ...)*, void (%struct.C*, ...)** [[VPTR]]
// CHECK64: musttail call void (%struct.C*, ...) [[CALLEE]](%struct.C* %{{.*}}, ...)
// CHECK64-NEXT: ret void
// CHECK64: }

// Thunk for calling the virtual function in internal class D.
// CHECK32-LABEL: define internal x86_thiscallcc void @"??_9D@?A0x{{[^@]*}}@@$BA@AE"(%"struct.(anonymous namespace)::D"* %this, ...)
// CHECK32: #[[ATTR]]
// CHECK32: [[VPTR:%.*]] = getelementptr inbounds void (%"struct.(anonymous namespace)::D"*, ...)*, void (%"struct.(anonymous namespace)::D"*, ...)** %{{.*}}, i64 0
// CHECK32: [[CALLEE:%.*]] = load void (%"struct.(anonymous namespace)::D"*, ...)*, void (%"struct.(anonymous namespace)::D"*, ...)** [[VPTR]]
// CHECK32: musttail call x86_thiscallcc void (%"struct.(anonymous namespace)::D"*, ...) [[CALLEE]](%"struct.(anonymous namespace)::D"* %{{.*}}, ...)
// CHECK32-NEXT: ret void
// CHECK32: }
//
// CHECK64-LABEL: define internal void @"??_9D@?A0x{{[^@]*}}@@$BA@AA"(%"struct.(anonymous namespace)::D"* %this, ...)
// CHECK64: #[[ATTR]]
// CHECK64: [[VPTR:%.*]] = getelementptr inbounds void (%"struct.(anonymous namespace)::D"*, ...)*, void (%"struct.(anonymous namespace)::D"*, ...)** %{{.*}}, i64 0
// CHECK64: [[CALLEE:%.*]] = load void (%"struct.(anonymous namespace)::D"*, ...)*, void (%"struct.(anonymous namespace)::D"*, ...)** [[VPTR]]
// CHECK64: musttail call void (%"struct.(anonymous namespace)::D"*, ...) [[CALLEE]](%"struct.(anonymous namespace)::D"* %{{.*}}, ...)
// CHECK64-NEXT: ret void
// CHECK64: }

// Thunk for calling the fourth virtual function in C, taking a struct parameter
// and returning a struct.
// CHECK32-LABEL: define linkonce_odr x86_thiscallcc void @"??_9C@@$BM@AE"(%struct.C* %this, ...) {{.*}} comdat
// CHECK32: [[VPTR:%.*]] = getelementptr inbounds void (%struct.C*, ...)*, void (%struct.C*, ...)** %{{.*}}, i64 3
// CHECK32: [[CALLEE:%.*]] = load void (%struct.C*, ...)*, void (%struct.C*, ...)** [[VPTR]]
// CHECK32: musttail call x86_thiscallcc void (%struct.C*, ...) [[CALLEE]](%struct.C* %{{.*}}, ...)
// CHECK32-NEXT: ret void
// CHECK32: }
//
// CHECK64-LABEL: define linkonce_odr void @"??_9C@@$BBI@AA"(%struct.C* %this, ...) {{.*}} comdat
// CHECK64: [[VPTR:%.*]] = getelementptr inbounds void (%struct.C*, ...)*, void (%struct.C*, ...)** %{{.*}}, i64 3
// CHECK64: [[CALLEE:%.*]] = load void (%struct.C*, ...)*, void (%struct.C*, ...)** [[VPTR]]
// CHECK64: musttail call void (%struct.C*, ...) [[CALLEE]](%struct.C* %{{.*}}, ...)
// CHECK64: ret void
// CHECK64: }

// Thunk for calling the fifth virtual function in C which uses the __cdecl calling convention.
// CHECK32-LABEL: define linkonce_odr void @"??_9C@@$BBA@AA"(%struct.C* %this, ...) {{.*}} comdat align 2 {
// CHECK32: [[VPTR:%.*]] = getelementptr inbounds void (%struct.C*, ...)*, void (%struct.C*, ...)** %{{.*}}, i64 4
// CHECK32: [[CALLEE:%.*]] = load void (%struct.C*, ...)*, void (%struct.C*, ...)** [[VPTR]]
// CHECK32: musttail call void (%struct.C*, ...) [[CALLEE]](%struct.C* %{{.*}}, ...)
// CHECK32: ret void
// CHECK32: }
//
// CHECK64-LABEL: define linkonce_odr void @"??_9C@@$BCA@AA"(%struct.C* %this, ...) {{.*}} comdat align 2 {
// CHECK64: [[VPTR:%.*]] = getelementptr inbounds void (%struct.C*, ...)*, void (%struct.C*, ...)** %{{.*}}, i64 4
// CHECK64: [[CALLEE:%.*]] = load void (%struct.C*, ...)*, void (%struct.C*, ...)** [[VPTR]]
// CHECK64: musttail call void (%struct.C*, ...) [[CALLEE]](%struct.C* %{{.*}}, ...)
// CHECK64: ret void
// CHECK64: }

// CHECK32: define linkonce_odr x86_thiscallcc void @"??_9C@@$BBE@AE"(%struct.C* %this, ...) {{.*}} comdat align 2 {
// CHECK32: [[VPTR:%.*]] = getelementptr inbounds void (%struct.C*, ...)*, void (%struct.C*, ...)** %{{.*}}, i64 5
// CHECK32: [[CALLEE:%.*]] = load void (%struct.C*, ...)*, void (%struct.C*, ...)** [[VPTR]]
// CHECK32: musttail call x86_thiscallcc void (%struct.C*, ...) [[CALLEE]](%struct.C* %{{.*}}, ...)
// CHECK32: ret void
// CHECK32: }

// CHECK64: define linkonce_odr void @"??_9C@@$BCI@AA"(%struct.C* %this, ...) {{.*}} comdat align 2 {
// CHECK64: [[VPTR:%.*]] = getelementptr inbounds void (%struct.C*, ...)*, void (%struct.C*, ...)** %{{.*}}, i64 5
// CHECK64: [[CALLEE:%.*]] = load void (%struct.C*, ...)*, void (%struct.C*, ...)** [[VPTR]]
// CHECK64: musttail call void (%struct.C*, ...) [[CALLEE]](%struct.C* %{{.*}}, ...)
// CHECK64: ret void
// CHECK64: }

// CHECK32: #[[ATTR]] = {{{.*}}"thunk"{{.*}}}

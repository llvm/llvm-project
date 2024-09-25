// RUN: %clang_cc1 -triple arm64-apple-ios -std=c++98 -fptrauth-calls -fapple-kext -fno-rtti -disable-O0-optnone -emit-llvm -o - %s | FileCheck %s

// CHECK: @_ZTV5TemplIiE = internal unnamed_addr constant { [7 x ptr] } { [7 x ptr] [ptr null, ptr null, ptr ptrauth (ptr @_ZN5TemplIiED1Ev, i32 0, i64 57986, ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV5TemplIiE, i32 0, i32 0, i32 2)), ptr ptrauth (ptr @_ZN5TemplIiED0Ev, i32 0, i64 22856, ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV5TemplIiE, i32 0, i32 0, i32 3)), ptr ptrauth (ptr @_ZN5TemplIiE1fEv, i32 0, i64 22189, ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV5TemplIiE, i32 0, i32 0, i32 4)), ptr ptrauth (ptr @_ZN5TemplIiE1gEv, i32 0, i64 9912, ptr getelementptr inbounds ({ [7 x ptr] }, ptr @_ZTV5TemplIiE, i32 0, i32 0, i32 5)), ptr null] }, align 8

struct B1 {
  virtual ~B1();
};

B1::~B1() {}

void DELETE(B1 *pb1) {
  pb1->B1::~B1();
}
// CHECK-LABEL: define void @_ZN2B1D0Ev
// CHECK: [[T1:%.*]] = load ptr, ptr getelementptr inbounds (ptr, ptr @_ZTV2B1, i64 2)
// CHECK-NEXT: [[B1:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr getelementptr inbounds (ptr, ptr @_ZTV2B1, i64 2) to i64), i64 14635)
// CHECK-NEXT: call noundef ptr [[T1]](ptr noundef nonnull align 8 dereferenceable(8) [[T2:%.*]]) [ "ptrauth"(i32 0, i64 [[B1]]) ]
// CHECK-LABEL: define void @_Z6DELETEP2B1
// CHECK: [[T3:%.*]] = load ptr, ptr getelementptr inbounds (ptr, ptr @_ZTV2B1, i64 2)
// CHECK-NEXT: [[B3:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr getelementptr inbounds (ptr, ptr @_ZTV2B1, i64 2) to i64), i64 14635)
// CHECK-NEXT:  call noundef ptr [[T3]](ptr noundef nonnull align 8 dereferenceable(8) [[T4:%.*]]) [ "ptrauth"(i32 0, i64 [[B3]])

template<class T>
struct Templ {
  virtual ~Templ(); // Out-of-line so that the destructor doesn't cause a vtable
  virtual void f() {}
  virtual void g() {}
};
template<class T>
struct SubTempl : public Templ<T> {
  virtual ~SubTempl() {} // override
  virtual void f() {} // override
  virtual void g() {} // override
};

void f(SubTempl<int>* t) {
  // Qualified calls go through the (qualified) vtable in apple-kext mode.
  // Since t's this pointer points to SubTempl's vtable, the call needs
  // to load Templ<int>'s vtable.  Hence, Templ<int>::g needs to be
  // instantiated in this TU, for it's referenced by the vtable.
  // (This happens only in apple-kext mode; elsewhere virtual calls can always
  // use the vtable pointer off this instead of having to load the vtable
  // symbol.)
  t->Templ::~Templ();
}

// CHECK: getelementptr inbounds (ptr, ptr @_ZTV5TemplIiE, i64 2)
// CHECK: declare void @_ZN5TemplIiED0Ev(ptr noundef nonnull align 8 dereferenceable(8))
// CHECK: define internal void @_ZN5TemplIiE1fEv(ptr noundef nonnull align 8 dereferenceable(8) %this)
// CHECK: define internal void @_ZN5TemplIiE1gEv(ptr noundef nonnull align 8 dereferenceable(8) %this)

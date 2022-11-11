// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fapple-kext -fno-rtti -disable-O0-optnone -emit-llvm -o - %s | FileCheck %s

// CHECK: @_ZTV5TemplIiE = internal unnamed_addr constant { [7 x ptr] } { [7 x ptr] [ptr null, ptr null, ptr @_ZN5TemplIiED1Ev, ptr @_ZN5TemplIiED0Ev, ptr @_ZN5TemplIiE1fEv, ptr @_ZN5TemplIiE1gEv, ptr null] }

struct B1 { 
  virtual ~B1(); 
};

B1::~B1() {}

void DELETE(B1 *pb1) {
  pb1->B1::~B1();
}
// CHECK-LABEL: define{{.*}} void @_ZN2B1D0Ev
// CHECK: [[T1:%.*]] = load ptr, ptr getelementptr inbounds (ptr, ptr @_ZTV2B1, i64 2)
// CHECK-NEXT: call void [[T1]](ptr {{[^,]*}} [[T2:%.*]])
// CHECK-LABEL: define{{.*}} void @_Z6DELETEP2B1
// CHECK: [[T3:%.*]] = load ptr, ptr getelementptr inbounds (ptr, ptr @_ZTV2B1, i64 2)
// CHECK-NEXT:  call void [[T3]](ptr {{[^,]*}} [[T4:%.*]])

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
// CHECK: declare void @_ZN5TemplIiED0Ev(ptr {{[^,]*}})
// CHECK: define internal void @_ZN5TemplIiE1fEv(ptr {{[^,]*}} %this)
// CHECK: define internal void @_ZN5TemplIiE1gEv(ptr {{[^,]*}} %this)

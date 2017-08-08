// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fapple-kext -emit-llvm -o - %s | FileCheck %s

// CHECK: @_ZTV5TemplIiE = internal unnamed_addr constant { [5 x i8*] } { [5 x i8*] [i8* null, i8* bitcast ({ i8*, i8* }* @_ZTI5TemplIiE to i8*), i8* bitcast (void (%struct.Templ*)* @_ZN5TemplIiE1fEv to i8*), i8* bitcast (void (%struct.Templ*)* @_ZN5TemplIiE1gEv to i8*), i8* null] }

struct Base { 
  virtual void abc(void) const; 
};

void Base::abc(void) const {}

void FUNC(Base* p) {
  p->Base::abc();
}

// CHECK: getelementptr inbounds (void (%struct.Base*)*, void (%struct.Base*)** bitcast ({ [4 x i8*] }* @_ZTV4Base to void (%struct.Base*)**), i64 2)
// CHECK-NOT: call void @_ZNK4Base3abcEv

template<class T>
struct Templ {
  virtual void f() {}
  virtual void g() {}
};
template<class T>
struct SubTempl : public Templ<T> {
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
  t->Templ::f();
}

// CHECK: getelementptr inbounds (void (%struct.Templ*)*, void (%struct.Templ*)** bitcast ({ [5 x i8*] }* @_ZTV5TemplIiE to void (%struct.Templ*)**), i64 2)
// CHECK: define internal void @_ZN5TemplIiE1fEv(%struct.Templ* %this)
// CHECK: define internal void @_ZN5TemplIiE1gEv(%struct.Templ* %this)

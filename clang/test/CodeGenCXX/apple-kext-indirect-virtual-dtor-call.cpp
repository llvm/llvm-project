// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fapple-kext -fno-rtti -disable-O0-optnone -emit-llvm -o - %s | FileCheck %s

// CHECK: @_ZTV5TemplIiE = internal unnamed_addr constant { [7 x i8*] } { [7 x i8*] [i8* null, i8* null, i8* bitcast (void (%struct.Templ*)* @_ZN5TemplIiED1Ev to i8*), i8* bitcast (void (%struct.Templ*)* @_ZN5TemplIiED0Ev to i8*), i8* bitcast (void (%struct.Templ*)* @_ZN5TemplIiE1fEv to i8*), i8* bitcast (void (%struct.Templ*)* @_ZN5TemplIiE1gEv to i8*), i8* null] }

struct B1 { 
  virtual ~B1(); 
};

B1::~B1() {}

void DELETE(B1 *pb1) {
  pb1->B1::~B1();
}
// CHECK-LABEL: define void @_ZN2B1D0Ev
// CHECK: [[T1:%.*]] = load void (%struct.B1*)*, void (%struct.B1*)** getelementptr inbounds (void (%struct.B1*)*, void (%struct.B1*)** bitcast ({ [5 x i8*] }* @_ZTV2B1 to void (%struct.B1*)**), i64 2)
// CHECK-NEXT: call void [[T1]](%struct.B1* [[T2:%.*]])
// CHECK-LABEL: define void @_Z6DELETEP2B1
// CHECK: [[T3:%.*]] = load void (%struct.B1*)*, void (%struct.B1*)** getelementptr inbounds (void (%struct.B1*)*, void (%struct.B1*)** bitcast ({ [5 x i8*] }* @_ZTV2B1 to void (%struct.B1*)**), i64 2)
// CHECK-NEXT:  call void [[T3]](%struct.B1* [[T4:%.*]])

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

// CHECK: getelementptr inbounds (void (%struct.Templ*)*, void (%struct.Templ*)** bitcast ({ [7 x i8*] }* @_ZTV5TemplIiE to void (%struct.Templ*)**), i64 2)
// CHECK: declare void @_ZN5TemplIiED0Ev(%struct.Templ*)
// CHECK: define internal void @_ZN5TemplIiE1fEv(%struct.Templ* %this)
// CHECK: define internal void @_ZN5TemplIiE1gEv(%struct.Templ* %this)

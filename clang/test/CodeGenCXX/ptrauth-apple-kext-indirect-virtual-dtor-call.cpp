// RUN: %clang_cc1 -triple arm64-apple-ios -std=c++98 -fptrauth-calls -fapple-kext -fno-rtti -disable-O0-optnone -emit-llvm -o - %s | FileCheck %s

// CHECK: @_ZTV5TemplIiE = internal unnamed_addr constant { [7 x i8*] } { [7 x i8*] [i8* null, i8* null, i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN5TemplIiED1Ev.ptrauth to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN5TemplIiED0Ev.ptrauth to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN5TemplIiE1fEv.ptrauth to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN5TemplIiE1gEv.ptrauth to i8*), i8* null] }

struct B1 {
  virtual ~B1();
};

B1::~B1() {}

void DELETE(B1 *pb1) {
  pb1->B1::~B1();
}
// CHECK-LABEL: define void @_ZN2B1D0Ev
// CHECK: [[T1:%.*]] = load %struct.B1* (%struct.B1*)*, %struct.B1* (%struct.B1*)** getelementptr inbounds (%struct.B1* (%struct.B1*)*, %struct.B1* (%struct.B1*)** bitcast ({ [5 x i8*] }* @_ZTV2B1 to %struct.B1* (%struct.B1*)**), i64 2)
// CHECK-NEXT: [[B1:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 ptrtoint (%struct.B1* (%struct.B1*)** getelementptr inbounds (%struct.B1* (%struct.B1*)*, %struct.B1* (%struct.B1*)** bitcast ({ [5 x i8*] }* @_ZTV2B1 to %struct.B1* (%struct.B1*)**), i64 2) to i64), i64 14635)
// CHECK-NEXT: call %struct.B1* [[T1]](%struct.B1* [[T2:%.*]]) [ "ptrauth"(i32 0, i64 [[B1]]) ]
// CHECK-LABEL: define void @_Z6DELETEP2B1
// CHECK: [[T3:%.*]] = load %struct.B1* (%struct.B1*)*, %struct.B1* (%struct.B1*)** getelementptr inbounds (%struct.B1* (%struct.B1*)*, %struct.B1* (%struct.B1*)** bitcast ({ [5 x i8*] }* @_ZTV2B1 to %struct.B1* (%struct.B1*)**), i64 2)
// CHECK-NEXT: [[B3:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 ptrtoint (%struct.B1* (%struct.B1*)** getelementptr inbounds (%struct.B1* (%struct.B1*)*, %struct.B1* (%struct.B1*)** bitcast ({ [5 x i8*] }* @_ZTV2B1 to %struct.B1* (%struct.B1*)**), i64 2) to i64), i64 14635)
// CHECK-NEXT:  call %struct.B1* [[T3]](%struct.B1* [[T4:%.*]]) [ "ptrauth"(i32 0, i64 [[B3]])

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

// CHECK: getelementptr inbounds (%struct.Templ* (%struct.Templ*)*, %struct.Templ* (%struct.Templ*)** bitcast ({ [7 x i8*] }* @_ZTV5TemplIiE to %struct.Templ* (%struct.Templ*)**), i64 2)
// CHECK: declare void @_ZN5TemplIiED0Ev(%struct.Templ*)
// CHECK: define internal void @_ZN5TemplIiE1fEv(%struct.Templ* %this)
// CHECK: define internal void @_ZN5TemplIiE1gEv(%struct.Templ* %this)

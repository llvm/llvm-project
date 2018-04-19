// RUN: %clang_cc1 -fno-rtti-data -triple x86_64-windows-msvc -emit-llvm %s -o - | FileCheck %s

// In this example, C does not override B::foo, but it needs to emit a thunk to
// adjust for the relative difference of position between A-in-B and A-in-C.

struct Incomplete;
template <typename T>
struct DoNotInstantiate {
  typename T::does_not_exist field;
};
template <typename T>
struct InstantiateLater;

struct A {
  virtual void foo(Incomplete p) = 0;
  virtual void bar(DoNotInstantiate<int> p) = 0;
  virtual int baz(InstantiateLater<int> p) = 0;
};
struct B : virtual A {
  void foo(Incomplete p) override;
  void bar(DoNotInstantiate<int> p) override;
  inline int baz(InstantiateLater<int> p) override;
};
struct C : B { int c; };
C c;

// Do the same thing, but with an incomplete return type.
struct B1 { virtual DoNotInstantiate<void> f() = 0; };
struct B2 { virtual DoNotInstantiate<void> f() = 0; };
struct S : B1, B2 { DoNotInstantiate<void> f() override; };
S s;

// CHECK: @"??_7S@@6BB2@@@" = linkonce_odr unnamed_addr constant
// CHECK-SAME: void (%struct.S*, ...)* @"?f@S@@W7EAA?AU?$DoNotInstantiate@X@@XZ"

// CHECK: @"??_7C@@6B@" = linkonce_odr unnamed_addr constant
// CHECK-SAME: void (%struct.B*, ...)* @"?foo@B@@W7EAAXUIncomplete@@@Z"
// CHECK-SAME: void (%struct.B*, ...)* @"?bar@B@@W7EAAXU?$DoNotInstantiate@H@@@Z"
// CHECK-SAME: i32 (i8*, i32)* @"?baz@B@@W7EAAHU?$InstantiateLater@H@@@Z"


// CHECK-LABEL: define linkonce_odr dso_local void @"?f@S@@W7EAA?AU?$DoNotInstantiate@X@@XZ"(%struct.S* %this, ...)
// CHECK: %[[THIS_ADJ_i8:[^ ]*]] = getelementptr i8, i8* {{.*}}, i32 -8
// CHECK: %[[THIS_ADJ:[^ ]*]] = bitcast i8* %[[THIS_ADJ_i8]] to %struct.S*
// CHECK: musttail call void (%struct.S*, ...) {{.*}}@"?f@S@@UEAA?AU?$DoNotInstantiate@X@@XZ"
// CHECK-SAME: (%struct.S* %[[THIS_ADJ]], ...)
// CHECK: ret void

// The thunks should have a -8 adjustment.

// CHECK-LABEL: define linkonce_odr dso_local void @"?foo@B@@W7EAAXUIncomplete@@@Z"(%struct.B* %this, ...)
// CHECK: %[[THIS_ADJ_i8:[^ ]*]] = getelementptr i8, i8* {{.*}}, i32 -8
// CHECK: %[[THIS_ADJ:[^ ]*]] = bitcast i8* %[[THIS_ADJ_i8]] to %struct.B*
// CHECK: musttail call void (%struct.B*, ...) {{.*}}@"?foo@B@@UEAAXUIncomplete@@@Z"
// CHECK-SAME: (%struct.B* %[[THIS_ADJ]], ...)
// CHECK-NEXT: ret void

// CHECK-LABEL: define linkonce_odr dso_local void @"?bar@B@@W7EAAXU?$DoNotInstantiate@H@@@Z"(%struct.B* %this, ...)
// CHECK: %[[THIS_ADJ_i8:[^ ]*]] = getelementptr i8, i8* {{.*}}, i32 -8
// CHECK: %[[THIS_ADJ:[^ ]*]] = bitcast i8* %[[THIS_ADJ_i8]] to %struct.B*
// CHECK: musttail call void (%struct.B*, ...) {{.*}}@"?bar@B@@UEAAXU?$DoNotInstantiate@H@@@Z"
// CHECK-SAME: (%struct.B* %[[THIS_ADJ]], ...)
// CHECK-NEXT: ret void

// If we complete the definition later, things work out.
template <typename T> struct InstantiateLater { T x; };
inline int B::baz(InstantiateLater<int> p) { return p.x; }

// CHECK-LABEL: define linkonce_odr dso_local i32 @"?baz@B@@W7EAAHU?$InstantiateLater@H@@@Z"(i8* %this.coerce, i32 %p.coerce)
// CHECK: = getelementptr i8, i8* {{.*}}, i32 -8
// CHECK: tail call i32 @"?baz@B@@UEAAHU?$InstantiateLater@H@@@Z"(i8* {{[^,]*}}, i32 {{.*}})

// CHECK-LABEL: define linkonce_odr dso_local i32 @"?baz@B@@UEAAHU?$InstantiateLater@H@@@Z"(i8* %this.coerce, i32 %p.coerce)

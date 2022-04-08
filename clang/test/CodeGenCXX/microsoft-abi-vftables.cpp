// RUN: %clang_cc1 -no-opaque-pointers %s -fno-rtti -triple=i386-pc-win32 -fms-extensions -emit-llvm -o - -O1 -disable-llvm-passes | FileCheck %s -check-prefix=NO-RTTI
// RUN: %clang_cc1 -no-opaque-pointers %s -triple=i386-pc-win32 -fms-extensions -emit-llvm -o - -O1 -disable-llvm-passes | FileCheck %s -check-prefix=RTTI

// RTTI-DAG: $"??_7S@@6B@" = comdat largest
// RTTI-DAG: $"??_7V@@6B@" = comdat largest

struct S {
  virtual ~S();
} s;

// RTTI-DAG: [[VTABLE_S:@.*]] = private unnamed_addr constant { [2 x i8*] } { [2 x i8*] [i8* bitcast ({{.*}} @"??_R4S@@6B@" to i8*), i8* bitcast ({{.*}} @"??_GS@@UAEPAXI@Z" to i8*)] }, comdat($"??_7S@@6B@")
// RTTI-DAG: @"??_7S@@6B@" = unnamed_addr alias i8*, getelementptr inbounds ({ [2 x i8*] }, { [2 x i8*] }* [[VTABLE_S]], i32 0, i32 0, i32 1)

// NO-RTTI-DAG: @"??_7S@@6B@" = linkonce_odr unnamed_addr constant { [1 x i8*] } { [1 x i8*] [i8* bitcast ({{.*}} @"??_GS@@UAEPAXI@Z" to i8*)] }

struct __declspec(dllimport) U {
  virtual ~U();
} u;

// RTTI-DAG: [[VTABLE_U:@.*]] = private unnamed_addr constant { [2 x i8*] } { [2 x i8*] [i8* bitcast ({{.*}} @"??_R4U@@6B@" to i8*), i8* bitcast ({{.*}} @"??_GU@@UAEPAXI@Z" to i8*)] }
// RTTI-DAG: @"??_SU@@6B@" = unnamed_addr alias i8*, getelementptr inbounds ({ [2 x i8*] }, { [2 x i8*] }* [[VTABLE_U]], i32 0, i32 0, i32 1)

// NO-RTTI-DAG: @"??_SU@@6B@" = linkonce_odr unnamed_addr constant { [1 x i8*] } { [1 x i8*] [i8* bitcast ({{.*}} @"??_GU@@UAEPAXI@Z" to i8*)] }

struct __declspec(dllexport) V {
  virtual ~V();
} v;

// RTTI-DAG: [[VTABLE_V:@.*]] = private unnamed_addr constant { [2 x i8*] } { [2 x i8*] [i8* bitcast ({{.*}} @"??_R4V@@6B@" to i8*), i8* bitcast ({{.*}} @"??_GV@@UAEPAXI@Z" to i8*)] }, comdat($"??_7V@@6B@")
// RTTI-DAG: @"??_7V@@6B@" = dllexport unnamed_addr alias i8*, getelementptr inbounds ({ [2 x i8*] }, { [2 x i8*] }* [[VTABLE_V]], i32 0, i32 0, i32 1)

// NO-RTTI-DAG: @"??_7V@@6B@" = weak_odr dllexport unnamed_addr constant { [1 x i8*] } { [1 x i8*] [i8* bitcast ({{.*}} @"??_GV@@UAEPAXI@Z" to i8*)] }

namespace {
struct W {
  virtual ~W() {}
} w;
}
// RTTI-DAG: [[VTABLE_W:@.*]] = private unnamed_addr constant { [2 x i8*] } { [2 x i8*] [i8* bitcast ({{.*}} @"??_R4W@?A0x{{[^@]*}}@@6B@" to i8*), i8* bitcast ({{.*}} @"??_GW@?A0x{{[^@]*}}@@UAEPAXI@Z" to i8*)] }
// RTTI-DAG: @"??_7W@?A0x{{[^@]*}}@@6B@" = internal unnamed_addr alias i8*, getelementptr inbounds ({ [2 x i8*] }, { [2 x i8*] }* [[VTABLE_W]], i32 0, i32 0, i32 1)

// NO-RTTI-DAG: @"??_7W@?A0x{{[^@]*}}@@6B@" = internal unnamed_addr constant { [1 x i8*] } { [1 x i8*] [i8* bitcast ({{.*}} @"??_GW@?A0x{{[^@]*}}@@UAEPAXI@Z" to i8*)] }

struct X {};
template <class> struct Y : virtual X {
  Y() {}
  virtual ~Y();
};

extern template class Y<int>;
template Y<int>::Y();
// RTTI-DAG: [[VTABLE_Y:@.*]] = private unnamed_addr constant { [2 x i8*] } { [2 x i8*] [i8* bitcast (%rtti.CompleteObjectLocator* @"??_R4?$Y@H@@6B@" to i8*), i8* bitcast (i8* (%struct.Y*, i32)* @"??_G?$Y@H@@UAEPAXI@Z" to i8*)] }, comdat($"??_7?$Y@H@@6B@")
// RTTI-DAG: @"??_7?$Y@H@@6B@" = unnamed_addr alias i8*, getelementptr inbounds ({ [2 x i8*] }, { [2 x i8*] }* [[VTABLE_Y]], i32 0, i32 0, i32 1)

// NO-RTTI-DAG: @"??_7?$Y@H@@6B@" = linkonce_odr unnamed_addr constant { [1 x i8*] } { [1 x i8*] [i8* bitcast (i8* (%struct.Y*, i32)* @"??_G?$Y@H@@UAEPAXI@Z" to i8*)] }, comdat

// RUN: %clang_cc1 -no-opaque-pointers -triple i686-windows-pc -emit-llvm -o - -mconstructor-aliases -O1 -disable-llvm-passes %s | FileCheck %s

struct Base { virtual ~Base(); };
struct Derived : Base {
  virtual ~Derived();
  static Derived inst;
};

Base::~Base(){}
Derived::~Derived(){}
Derived Derived::inst;

// CHECK: @"??1Derived@@UAE@XZ" = dso_local unnamed_addr alias void (%struct.Derived*), bitcast (void (%struct.Base*)* @"??1Base@@UAE@XZ" to void (%struct.Derived*)*)

// CHECK: define dso_local x86_thiscallcc void @"??1Base@@UAE@XZ"
// CHECK: define internal void @"??__E?inst@Derived@@2U1@A@@YAXXZ"
// CHECK: call i32 @atexit(void ()* @"??__F?inst@Derived@@2U1@A@@YAXXZ"
//
// CHECK: define internal void @"??__F?inst@Derived@@2U1@A@@YAXXZ"
// CHECK-NEXT: entry:
// CHECK-NEXT: call x86_thiscallcc void @"??1Derived@@UAE@XZ"

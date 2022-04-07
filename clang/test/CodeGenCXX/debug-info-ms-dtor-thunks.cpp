// RUN: %clang_cc1 -no-opaque-pointers -triple i686--windows -emit-llvm -debug-info-kind=line-tables-only -x c++ %s -fms-extensions -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -triple i686--windows -emit-llvm -debug-info-kind=line-directives-only -x c++ %s -fms-extensions -o - | FileCheck %s

struct __declspec(dllexport) S { virtual ~S(); };
struct __declspec(dllexport) T { virtual ~T(); };
struct __declspec(dllexport) U : S, T { virtual ~U(); };

// CHECK-LABEL: define {{.*}} @"??_GS@@UAEPAXI@Z"
// CHECK: call x86_thiscallcc void @"??1S@@UAE@XZ"(%struct.S* {{[^,]*}} %this1){{.*}}!dbg !{{[0-9]+}}

// CHECK-LABEL: define {{.*}} @"??_GT@@UAEPAXI@Z"
// CHECK: call x86_thiscallcc void @"??1T@@UAE@XZ"(%struct.T* {{[^,]*}} %this1){{.*}}!dbg !{{[0-9]+}}

// CHECK-LABEL: define {{.*}} @"??_GU@@UAEPAXI@Z"
// CHECK: call x86_thiscallcc void @"??1U@@UAE@XZ"(%struct.U* {{[^,]*}} %this1){{.*}}!dbg !{{[0-9]+}}

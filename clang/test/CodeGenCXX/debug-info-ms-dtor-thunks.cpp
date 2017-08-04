// RUN: %clang_cc1 -triple i686--windows -emit-llvm -debug-info-kind=line-tables-only -x c++ %s -fms-extensions -o - | FileCheck %s

struct __declspec(dllexport) S { virtual ~S(); };
struct __declspec(dllexport) T { virtual ~T(); };
struct __declspec(dllexport) U : S, T { virtual ~U(); };

// CHECK-LABEL: define {{.*}} @"\01??_GS@@UAEPAXI@Z"
// CHECK: call x86_thiscallcc void @"\01??_DS@@QAEXXZ"(%struct.S* %this1){{.*}}!dbg !{{[0-9]+}}

// CHECK-LABEL: define {{.*}} @"\01??_GT@@UAEPAXI@Z"
// CHECK: call x86_thiscallcc void @"\01??_DT@@QAEXXZ"(%struct.T* %this1){{.*}}!dbg !{{[0-9]+}}

// CHECK-LABEL: define {{.*}} @"\01??_GU@@UAEPAXI@Z"
// CHECK: call x86_thiscallcc void @"\01??_DU@@QAEXXZ"(%struct.U* %this1){{.*}}!dbg !{{[0-9]+}}

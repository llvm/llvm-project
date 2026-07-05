// RUN: %clang_cc1 -triple i686-windows-msvc -fms-extensions -fno-rtti -emit-llvm -std=c++1y -O0 -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-windows-msvc -fms-extensions -fno-rtti -emit-llvm -std=c++1y -O0 -o - %s | FileCheck %s --check-prefix X64

extern int __declspec(dllimport) var;
extern void __declspec(dllimport) fun();

extern int *varp;
int *varp = &var;
// CHECK-DAG: @"?varp@@3PAHA" = dso_local global ptr null
// X64-DAG: @"?varp@@3PEAHEA" = dso_local global ptr null

extern void (*funp)();
void (*funp)() = &fun;
// CHECK-DAG: @"?funp@@3P6AXXZA" = dso_local global ptr null
// X64-DAG: @"?funp@@3P6AXXZEA" = dso_local global ptr null

// CHECK-LABEL: @"??__Evarp@@YAXXZ"
// CHECK-DAG: store ptr @"?var@@3HA", ptr @"?varp@@3PAHA"

// X64-LABEL: @"??__Evarp@@YAXXZ"
// X64-DAG: store ptr @"?var@@3HA", ptr @"?varp@@3PEAHEA"

// CHECK-LABEL: @"??__Efunp@@YAXXZ"()
// CHECK-DAG: store ptr @"?fun@@YAXXZ", ptr @"?funp@@3P6AXXZA"

// X64-LABEL: @"??__Efunp@@YAXXZ"()
// X64-DAG: store ptr @"?fun@@YAXXZ", ptr @"?funp@@3P6AXXZEA"

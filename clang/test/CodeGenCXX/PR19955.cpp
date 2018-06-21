// RUN: %clang_cc1 -triple i686-windows-msvc -fms-extensions -fno-rtti -emit-llvm -std=c++1y -O0 -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-windows-msvc -fms-extensions -fno-rtti -emit-llvm -std=c++1y -O0 -o - %s | FileCheck %s --check-prefix X64

extern int __declspec(dllimport) var;
extern void __declspec(dllimport) fun();

extern int *varp;
int *varp = &var;
// CHECK-DAG: @"\01?varp@@3PAHA" = global i32* null
// X64-DAG: @"\01?varp@@3PEAHEA" = global i32* null

extern void (*funp)();
void (*funp)() = &fun;
// CHECK-DAG: @"\01?funp@@3P6AXXZA" = global void ()* null
// X64-DAG: @"\01?funp@@3P6AXXZEA" = global void ()* null

// CHECK-LABEL: @"\01??__Evarp@@YAXXZ"
// CHECK-DAG: store i32* @"\01?var@@3HA", i32** @"\01?varp@@3PAHA"

// X64-LABEL: @"\01??__Evarp@@YAXXZ"
// X64-DAG: store i32* @"\01?var@@3HA", i32** @"\01?varp@@3PEAHEA"

// CHECK-LABEL: @"\01??__Efunp@@YAXXZ"()
// CHECK-DAG: store void ()* @"\01?fun@@YAXXZ", void ()** @"\01?funp@@3P6AXXZA"

// X64-LABEL: @"\01??__Efunp@@YAXXZ"()
// X64-DAG: store void ()* @"\01?fun@@YAXXZ", void ()** @"\01?funp@@3P6AXXZEA"

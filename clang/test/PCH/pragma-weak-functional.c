// Test this without pch.
// RUN: %clang_cc1 -no-opaque-pointers -include %S/pragma-weak-functional.h %s -verify -emit-llvm -o - | FileCheck %s

// Test with pch.
// RUN: %clang_cc1 -no-opaque-pointers -x c-header -emit-pch -o %t %S/pragma-weak-functional.h
// RUN: %clang_cc1 -no-opaque-pointers -include-pch %t %s -verify -emit-llvm -o - | FileCheck %s

// CHECK-DAG: @undecfunc_alias1 = weak{{.*}} alias void (), void ()* @undecfunc
// CHECK-DAG: @undecfunc_alias2 = weak{{.*}} alias void (), void ()* @undecfunc
// CHECK-DAG: @undecfunc_alias3 = weak{{.*}} alias void (), void ()* @undecfunc
// CHECK-DAG: @undecfunc_alias4 = weak{{.*}} alias void (), void ()* @undecfunc

///////////// PR28611: Try multiple aliases of same undeclared symbol or alias
void undecfunc_alias1(void);
void undecfunc(void) { }
// expected-warning@pragma-weak-functional.h:4 {{alias will always resolve to undecfunc}}
// expected-warning@pragma-weak-functional.h:5 {{alias will always resolve to undecfunc}}

// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s

int &f();

// CHECK: @r = thread_local global i32* null
thread_local int &r = f();

// CHECK: @_ZTH1r = alias void (), void ()* @__tls_init

int &g() { return r; }

// CHECK: define {{.*}} @[[R_INIT:.*]]()
// CHECK: call dereferenceable({{[0-9]+}}) i32* @_Z1fv()
// CHECK: store i32* %{{.*}}, i32** @r, align 8

// CHECK-LABEL: define dereferenceable({{[0-9]+}}) i32* @_Z1gv()
// CHECK: call i32* @_ZTW1r()
// CHECK: ret i32* %{{.*}}

// CHECK: define weak_odr hidden i32* @_ZTW1r() {
// CHECK: call void @_ZTH1r()
// CHECK: load i32*, i32** @r, align 8
// CHECK: ret i32* %{{.*}}

// CHECK-LABEL: define internal void @__tls_init()
// CHECK: call void @[[R_INIT]]()

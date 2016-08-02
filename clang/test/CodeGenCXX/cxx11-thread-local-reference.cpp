// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck --check-prefix=CHECK --check-prefix=LINUX %s
// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -triple x86_64-apple-darwin12 | FileCheck --check-prefix=CHECK --check-prefix=DARWIN %s

int &f();

// LINUX: @r = thread_local global i32* null
// DARWIN: @r = internal thread_local global i32* null
thread_local int &r = f();

// LINUX: @_ZTH1r = alias void (), void ()* @__tls_init
// DARWIN: @_ZTH1r = internal alias void (), void ()* @__tls_init

int &g() { return r; }

// CHECK: define {{.*}} @[[R_INIT:.*]]()
// CHECK: call dereferenceable({{[0-9]+}}) i32* @_Z1fv()
// CHECK: store i32* %{{.*}}, i32** @r, align 8

// CHECK-LABEL: define dereferenceable({{[0-9]+}}) i32* @_Z1gv()
// LINUX: call i32* @_ZTW1r()
// DARWIN: call cxx_fast_tlscc i32* @_ZTW1r()
// CHECK: ret i32* %{{.*}}

// LINUX: define weak_odr hidden i32* @_ZTW1r() [[ATTR0:#[0-9]+]] {
// DARWIN: define cxx_fast_tlscc i32* @_ZTW1r() [[ATTR1:#[0-9]+]] {
// CHECK: call void @_ZTH1r()
// CHECK: load i32*, i32** @r, align 8
// CHECK: ret i32* %{{.*}}

// CHECK-LABEL: define internal void @__tls_init()
// CHECK: call void @[[R_INIT]]()

// LINUX: attributes [[ATTR0]] = { {{.*}}"target-features"{{.*}} }
// DARWIN: attributes [[ATTR1]] = { {{.*}}nounwind{{.*}}"target-features"{{.*}}  }

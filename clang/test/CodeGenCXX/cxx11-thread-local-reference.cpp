// RUN: %clang_cc1 -no-opaque-pointers -std=c++11 -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck --check-prefixes=CHECK,LINUX_AIX %s
// RUN: %clang_cc1 -no-opaque-pointers -std=c++11 -emit-llvm %s -o - -triple powerpc64-unknown-aix-xcoff | FileCheck --check-prefixes=CHECK,LINUX_AIX %s
// RUN: %clang_cc1 -no-opaque-pointers -std=c++11 -emit-llvm %s -o - -triple x86_64-apple-darwin12 | FileCheck --check-prefix=CHECK --check-prefix=DARWIN %s

int &f();

// LINUX_AIX: @r ={{.*}} thread_local global i32* null
// DARWIN: @r = internal thread_local global i32* null
thread_local int &r = f();

// LINUX_AIX: @_ZTH1r ={{.*}} alias void (), void ()* @__tls_init
// DARWIN: @_ZTH1r = internal alias void (), void ()* @__tls_init

int &g() { return r; }

// CHECK: define {{.*}} @[[R_INIT:.*]]()
// CHECK: call noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) i32* @_Z1fv()
// CHECK: %[[R_ADDR:.+]] = call i32** @llvm.threadlocal.address.p0p0i32(i32** @r)
// CHECK: store i32* %{{.*}}, i32** %[[R_ADDR]], align 8

// CHECK-LABEL: define{{.*}} nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) i32* @_Z1gv()
// LINUX_AIX: call i32* @_ZTW1r()
// DARWIN: call cxx_fast_tlscc i32* @_ZTW1r()
// CHECK: ret i32* %{{.*}}

// LINUX_AIX: define weak_odr hidden noundef i32* @_ZTW1r() [[ATTR0:#[0-9]+]]{{( comdat)?}} {
// DARWIN: define cxx_fast_tlscc noundef i32* @_ZTW1r() [[ATTR1:#[0-9]+]] {
// LINUX_AIX: call void @_ZTH1r()
// DARWIN: call cxx_fast_tlscc void @_ZTH1r()
// CHECK: %[[R_ADDR2:.+]] = call i32** @llvm.threadlocal.address.p0p0i32(i32** @r)
// CHECK: load i32*, i32** %[[R_ADDR2]], align 8
// CHECK: ret i32* %{{.*}}

// LINUX_AIX-LABEL: define internal void @__tls_init()
// DARWIN-LABEL: define internal cxx_fast_tlscc void @__tls_init()
// CHECK: call void @[[R_INIT]]()

// LINUX_AIX: attributes [[ATTR0]] = { {{.*}}"target-features"{{.*}} }
// DARWIN: attributes [[ATTR1]] = { {{.*}}nounwind{{.*}}"target-features"{{.*}}  }

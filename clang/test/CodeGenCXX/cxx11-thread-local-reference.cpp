// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck --check-prefixes=CHECK,LINUX_AIX %s
// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -triple powerpc64-unknown-aix-xcoff | FileCheck --check-prefixes=CHECK,LINUX_AIX %s
// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -triple x86_64-apple-darwin12 | FileCheck --check-prefix=CHECK --check-prefix=DARWIN %s

int &f();

// LINUX_AIX: @r ={{.*}} thread_local global ptr null
// DARWIN: @r = internal thread_local global ptr null
thread_local int &r = f();

// LINUX_AIX: @_ZTH1r ={{.*}} alias void (), ptr @__tls_init
// DARWIN: @_ZTH1r = internal alias void (), ptr @__tls_init

int &g() { return r; }

// CHECK: define {{.*}} @[[R_INIT:.*]]()
// CHECK: call noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_Z1fv()
// CHECK: %[[R_ADDR:.+]] = call align 8 ptr @llvm.threadlocal.address.p0(ptr align 8 @r)
// CHECK: store ptr %{{.*}}, ptr %[[R_ADDR]], align 8

// CHECK-LABEL: define{{.*}} nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_Z1gv()
// LINUX_AIX: call ptr @_ZTW1r()
// DARWIN: call cxx_fast_tlscc ptr @_ZTW1r()
// CHECK: ret ptr %{{.*}}

// LINUX_AIX: define weak_odr hidden noundef ptr @_ZTW1r() [[ATTR0:#[0-9]+]]{{( comdat)?}} {
// DARWIN: define cxx_fast_tlscc noundef ptr @_ZTW1r() [[ATTR1:#[0-9]+]] {
// LINUX_AIX: call void @_ZTH1r()
// DARWIN: call cxx_fast_tlscc void @_ZTH1r()
// CHECK: %[[R_ADDR2:.+]] = call align 8 ptr @llvm.threadlocal.address.p0(ptr align 8 @r)
// CHECK: load ptr, ptr %[[R_ADDR2]], align 8
// CHECK: ret ptr %{{.*}}

// LINUX_AIX-LABEL: define internal void @__tls_init()
// DARWIN-LABEL: define internal cxx_fast_tlscc void @__tls_init()
// CHECK: call void @[[R_INIT]]()

// LINUX_AIX: attributes [[ATTR0]] = { {{.*}}"target-features"{{.*}} }
// DARWIN: attributes [[ATTR1]] = { {{.*}}nounwind{{.*}}"target-features"{{.*}}  }

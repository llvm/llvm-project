// RUN: %clang_cc1 %s -fxray-instrument -std=c++11 -x c++ -emit-llvm -o - -triple x86_64-unknown-linux-gnu | FileCheck %s

// Make sure that the LLVM attribute for XRay-annotated functions do show up.
[[clang::xray_always_instrument,clang::xray_log_args(1)]] void foo(int a) {
// CHECK: define{{.*}} void @_Z3fooi(i32 noundef %a) #0
};

[[clang::xray_log_args(1)]] void bar(int a) {
// CHECK: define{{.*}} void @_Z3bari(i32 noundef %a) #1
};

// CHECK: #0 = {{.*}}"function-instrument"="xray-always"{{.*}}"xray-log-args"="1"
// CHECK-NOT: #1 = {{.*}}"xray-log-args"="1"

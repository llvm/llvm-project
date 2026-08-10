// We want to ensure that the "never instrument" attributes show up even if we
// explicitly turn off XRay instrumentation.
//
/// -fno-xray-instrument is the default. It does not produce a CC1 option.
// RUN: %clang_cc1 %s -std=c++11 -x c++ -emit-llvm -o - \
// RUN:     -triple x86_64-unknown-linux-gnu | FileCheck %s
// RUN: %clang_cc1 %s -std=c++11 -x c++ -emit-llvm -o - \
// RUN:     -triple arm-unknown-linux-gnu -target-abi apcs-gnu | FileCheck %s
// RUN: %clang_cc1 %s -std=c++11 -x c++ -emit-llvm -o - \
// RUN:     -triple mips-unknown-linux-gnu | FileCheck %s
// RUN: %clang_cc1 %s -std=c++11 -x c++ -emit-llvm -o - \
// RUN:     -triple mipsel-unknown-linux-gnu | FileCheck %s
// RUN: %clang_cc1 %s -std=c++11 -x c++ -emit-llvm -o - \
// RUN:     -triple mips64-unknown-linux-gnu | FileCheck %s
// RUN: %clang_cc1 %s -std=c++11 -x c++ -emit-llvm -o - \
// RUN:     -triple mips64el-unknown-linux-gnu | FileCheck %s
// RUN: %clang_cc1 %s -std=c++11 -x c++ -emit-llvm -o - \
// RUN:     -triple powerpc64le-unknown-linux-gnu | FileCheck %s

[[clang::xray_always_instrument]] void foo() {
// CHECK: define{{.*}} void @_Z3foov() #0
}

[[clang::xray_never_instrument]] void bar() {
// CHECK: define{{.*}} void @_Z3barv() #1
}

// CHECK-NOT: #0 = {{.*}}"function-instrument"="xray-always"
// CHECK: #1 = {{.*}}"function-instrument"="xray-never"


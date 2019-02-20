// RUN: %clang_cc1 %s -fxray-instrument -std=c++11 -x c++ -emit-llvm -o - \
// RUN:     -triple x86_64-unknown-linux-gnu | FileCheck %s
// RUN: %clang_cc1 %s -fxray-instrument -std=c++11 -x c++ -emit-llvm -o - \
// RUN:     -triple arm-unknown-linux-gnu -target-abi apcs-gnu | FileCheck %s
// RUN: %clang_cc1 %s -fxray-instrument -std=c++11 -x c++ -emit-llvm -o - \
// RUN:     -triple mips-unknown-linux-gnu | FileCheck %s
// RUN: %clang_cc1 %s -fxray-instrument -std=c++11 -x c++ -emit-llvm -o - \
// RUN:     -triple mipsisa32r6-unknown-linux-gnu | FileCheck %s
// RUN: %clang_cc1 %s -fxray-instrument -std=c++11 -x c++ -emit-llvm -o - \
// RUN:     -triple mipsel-unknown-linux-gnu | FileCheck %s
// RUN: %clang_cc1 %s -fxray-instrument -std=c++11 -x c++ -emit-llvm -o - \
// RUN:     -triple mipsisa32r6el-unknown-linux-gnu | FileCheck %s
// RUN: %clang_cc1 %s -fxray-instrument -std=c++11 -x c++ -emit-llvm -o - \
// RUN:     -triple mips64-unknown-linux-gnu | FileCheck %s
// RUN: %clang_cc1 %s -fxray-instrument -std=c++11 -x c++ -emit-llvm -o - \
// RUN:     -triple mipsisa64r6-unknown-linux-gnu | FileCheck %s
// RUN: %clang_cc1 %s -fxray-instrument -std=c++11 -x c++ -emit-llvm -o - \
// RUN:     -triple mips64el-unknown-linux-gnu | FileCheck %s
// RUN: %clang_cc1 %s -fxray-instrument -std=c++11 -x c++ -emit-llvm -o - \
// RUN:     -triple mipsisa64r6el-unknown-linux-gnu | FileCheck %s
// RUN: %clang_cc1 %s -fxray-instrument -std=c++11 -x c++ -emit-llvm -o - \
// RUN:     -triple mips64-unknown-linux-gnuabi64 | FileCheck %s
// RUN: %clang_cc1 %s -fxray-instrument -std=c++11 -x c++ -emit-llvm -o - \
// RUN:     -triple mipsisa64r6-unknown-linux-gnuabi64 | FileCheck %s
// RUN: %clang_cc1 %s -fxray-instrument -std=c++11 -x c++ -emit-llvm -o - \
// RUN:     -triple mips64el-unknown-linux-gnuabi64 | FileCheck %s
// RUN: %clang_cc1 %s -fxray-instrument -std=c++11 -x c++ -emit-llvm -o - \
// RUN:     -triple mipsisa64r6el-unknown-linux-gnuabi64 | FileCheck %s
// RUN: %clang_cc1 %s -fxray-instrument -std=c++11 -x c++ -emit-llvm -o - \
// RUN:     -triple mips64-unknown-linux-gnuabin32 | FileCheck %s
// RUN: %clang_cc1 %s -fxray-instrument -std=c++11 -x c++ -emit-llvm -o - \
// RUN:     -triple mipsisa64r6-unknown-linux-gnuabin32 | FileCheck %s
// RUN: %clang_cc1 %s -fxray-instrument -std=c++11 -x c++ -emit-llvm -o - \
// RUN:     -triple mips64el-unknown-linux-gnuabin32 | FileCheck %s
// RUN: %clang_cc1 %s -fxray-instrument -std=c++11 -x c++ -emit-llvm -o - \
// RUN:     -triple mipsisa64r6el-unknown-linux-gnuabin32 | FileCheck %s
// RUN: %clang_cc1 %s -fxray-instrument -std=c++11 -x c++ -emit-llvm -o - \
// RUN:     -triple powerpc64le-unknown-linux-gnu | FileCheck %s

// Make sure that the LLVM attribute for XRay-annotated functions do show up.
[[clang::xray_always_instrument]] void foo() {
// CHECK: define void @_Z3foov() #0
};

[[clang::xray_never_instrument]] void bar() {
// CHECK: define void @_Z3barv() #1
};

// CHECK: #0 = {{.*}}"function-instrument"="xray-always"
// CHECK: #1 = {{.*}}"function-instrument"="xray-never"

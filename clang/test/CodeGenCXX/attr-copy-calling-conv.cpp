// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++23 -emit-llvm -o - %s | FileCheck %s

using Callback = int (__attribute__((ms_abi)) *)(int);
Callback callback = [](int x) __attribute__((copy((Callback)nullptr))) { return x; };

// CHECK: define internal win64cc noundef i32 {{.*}}__invoke{{.*}}(i32 noundef
// CHECK: call win64cc noundef i32 {{.*}}clEi
// CHECK: define internal win64cc noundef i32 {{.*}}clEi

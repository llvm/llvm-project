// RUN: %clang_cc1 -triple arm64e-apple-ios                   -emit-llvm %s  -o - | FileCheck %s --check-prefixes=ALL,OFF
// RUN: %clang_cc1 -triple aarch64-linux-gnu                  -emit-llvm %s  -o - | FileCheck %s --check-prefixes=ALL,OFF

// RUN: %clang_cc1 -triple arm64-apple-ios  -fptrauth-calls   -emit-llvm %s  -o - | FileCheck %s --check-prefixes=ALL,CALLS
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fptrauth-calls  -emit-llvm %s  -o - | FileCheck %s --check-prefixes=ALL,CALLS

// ALL: define {{(dso_local )?}}void @test() #0
void test() {
}

// CALLS: attributes #0 = {{{.*}} "ptrauth-calls" {{.*}}}

// OFF-NOT: attributes {{.*}} "ptrauth-

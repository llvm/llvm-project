// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir -target-cpu corei7 -target-feature +avx
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll -target-cpu corei7 -target-feature +avx
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll -target-cpu corei7 -target-feature +avx
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

int foo(int);

struct A {
  A(int);
};

A a(1);
__attribute__((init_priority(300))) A b(2);

// The default CPU and features are recorded once on the module.
// CIR: module {{.*}}"cir.target-cpu" = "corei7"{{.*}}"cir.target-features" = "{{.*}}+avx{{.*}}"

// CIR: cir.func internal private @__cxx_global_var_init() attributes {"cir.target-cpu" = "corei7", "cir.target-features" = "{{.*}}+avx{{.*}}"}
// CIR: cir.func internal private @__cxx_global_var_init.1() attributes {"cir.target-cpu" = "corei7", "cir.target-features" = "{{.*}}+avx{{.*}}"}
// CIR: cir.func internal private @_GLOBAL__I_000300() attributes {"cir.target-cpu" = "corei7", "cir.target-features" = "{{.*}}+avx{{.*}}"}
// CIR: cir.func internal private @_GLOBAL__sub_I_global_init_target_features.cpp() attributes {"cir.target-cpu" = "corei7", "cir.target-features" = "{{.*}}+avx{{.*}}"}

// LLVM: define internal void @__cxx_global_var_init(){{.*}} #[[ATTR:[0-9]+]]
// LLVM: define internal void @__cxx_global_var_init.1(){{.*}} #[[ATTR]]
// LLVM: define internal void @_GLOBAL__I_000300(){{.*}} #[[ATTR]]
// LLVM: define internal void @_GLOBAL__sub_I_global_init_target_features.cpp(){{.*}} #[[ATTR]]
// LLVM: attributes #[[ATTR]] = {{.*}}"target-cpu"="corei7"{{.*}}"target-features"="{{.*}}+avx{{.*}}"

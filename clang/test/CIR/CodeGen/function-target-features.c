// This test verifies that we produce cir.target-cpu and cir.target-features
// attributes on functions when they're different from the standard cpu and
// have written features.

// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fclangir -emit-cir -o - %s -target-feature +avx | FileCheck %s -check-prefix=AVX-FEATURE
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fclangir -emit-cir -o - %s -target-feature +avx | FileCheck %s -check-prefix=AVX-NO-CPU
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fclangir -emit-cir -o - %s -target-feature +avx512f -target-feature +avx512bw | FileCheck %s -check-prefix=TWO-AVX
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fclangir -emit-cir -o - %s -target-cpu corei7 | FileCheck %s -check-prefix=CORE-CPU
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fclangir -emit-cir -o - %s -target-cpu corei7 -target-feature +avx | FileCheck %s -check-prefix=CORE-CPU-AND-FEATURES
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fclangir -emit-cir -o - %s -target-cpu x86-64 | FileCheck %s -check-prefix=X86-64-CPU
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fclangir -emit-cir -o - %s -target-cpu corei7-avx -target-feature -avx | FileCheck %s -check-prefix=AVX-MINUS-FEATURE
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fclangir -emit-llvm -o - %s -target-cpu corei7 -target-feature +avx | FileCheck %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s -target-cpu corei7 -target-feature +avx | FileCheck %s -check-prefix=LLVM

void foo(void) {}

// AVX-FEATURE: "cir.target-features"{{.*}}+avx
// AVX-NO-CPU-NOT: cir.target-cpu
// TWO-AVX: "cir.target-features"{{.*}}+avx512bw{{.*}}+avx512f
// CORE-CPU: "cir.target-cpu" = "corei7"
// CORE-CPU-AND-FEATURES: "cir.target-cpu" = "corei7"
// CORE-CPU-AND-FEATURES-SAME: "cir.target-features"{{.*}}+avx
// X86-64-CPU: "cir.target-cpu" = "x86-64"
// AVX-MINUS-FEATURE: "cir.target-features"{{.*}}-avx

// LLVM: define{{.*}} void @foo(){{.*}} #[[ATTR:[0-9]+]]
// LLVM: attributes #[[ATTR]] = {{.*}}"target-cpu"="corei7"{{.*}}"target-features"={{.*}}+avx

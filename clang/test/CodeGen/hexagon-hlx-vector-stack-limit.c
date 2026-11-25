// RUN: %clang_cc1 -triple hexagon-unknown-elf -target-cpu hexagonv85 \
// RUN:   -target-feature +hlx -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple hexagon-unknown-elf -target-cpu hexagonv85 \
// RUN:   -target-feature +hlx -fhlx-vector-stack-limit=1024 -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=OPT

// LTO bitcode retains the attributes.
// RUN: %clang_cc1 -triple hexagon-unknown-elf -target-cpu hexagonv85 \
// RUN:   -target-feature +hlx -fhlx-vector-stack-limit=1024 -flto \
// RUN:   -emit-llvm-bc -o %t.bc %s
// RUN: llvm-dis %t.bc -o - | FileCheck %s --check-prefix=OPT
// RUN: %clang_cc1 -triple hexagon-unknown-elf -target-cpu hexagonv85 \
// RUN:   -target-feature +hlx -fhlx-vector-stack-limit=1024 -flto=thin \
// RUN:   -emit-llvm-bc -o %t.thin.bc %s
// RUN: llvm-dis %t.thin.bc -o - | FileCheck %s --check-prefix=OPT

// CHECK: define{{.*}} void @limited() [[ATTR:#[0-9]+]]
// CHECK: define{{.*}} void @zero() [[ZERO:#[0-9]+]]
// CHECK: define{{.*}} void @unlimited() [[NONE:#[0-9]+]]
// CHECK: attributes [[ATTR]] = {{{.*}}"hlx-vector-stack-limit"="8192"
// CHECK: attributes [[ZERO]] = {{{.*}}"hlx-vector-stack-limit"="0"
// CHECK-NOT: attributes [[NONE]] = {{{.*}}"hlx-vector-stack-limit"

// The function attribute overrides -fhlx-vector-stack-limit.
// OPT: define{{.*}} void @limited() [[ATTR:#[0-9]+]]
// OPT: define{{.*}} void @zero() [[ZERO:#[0-9]+]]
// OPT: define{{.*}} void @unlimited() [[NONE:#[0-9]+]]
// OPT: attributes [[ATTR]] = {{{.*}}"hlx-vector-stack-limit"="8192"
// OPT: attributes [[ZERO]] = {{{.*}}"hlx-vector-stack-limit"="0"
// OPT: attributes [[NONE]] = {{{.*}}"hlx-vector-stack-limit"="1024"

__attribute__((hlx_vector_stack_limit(8192))) void limited(void) {}
[[clang::hlx_vector_stack_limit(0)]] void zero(void) {}
void unlimited(void) {}

// REQUIRES: hexagon-registered-target
// RUN: %clang_cc1 -triple hexagon-unknown-elf -target-cpu hexagonv85 \
// RUN:   -target-feature +hlx -emit-llvm -o - %s | FileCheck %s

// CHECK: define{{.*}} void @limited() [[ATTR:#[0-9]+]]
// CHECK: define{{.*}} void @zero() [[ZERO:#[0-9]+]]
// CHECK: define{{.*}} void @unlimited() [[NONE:#[0-9]+]]
// CHECK: attributes [[ATTR]] = {{{.*}}"hlx-vector-stack-limit"="8192"
// CHECK: attributes [[ZERO]] = {{{.*}}"hlx-vector-stack-limit"="0"
// CHECK-NOT: attributes [[NONE]] = {{{.*}}"hlx-vector-stack-limit"

__attribute__((hlx_vector_stack_limit(8192))) void limited(void) {}
[[clang::hlx_vector_stack_limit(0)]] void zero(void) {}
void unlimited(void) {}

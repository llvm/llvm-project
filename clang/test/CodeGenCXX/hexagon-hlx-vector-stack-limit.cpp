// RUN: %clang_cc1 -triple hexagon-unknown-elf -target-cpu hexagonv85 \
// RUN:   -target-feature +hlx -fhlx-vector-stack-limit=1024 -emit-llvm -o - %s \
// RUN:   | FileCheck %s
// RUN: %clang_cc1 -triple hexagon-unknown-elf -target-cpu hexagonv85 \
// RUN:   -target-feature +hlx -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=NONE

// Functions generated without a FunctionDecl get the command-line limit.

int f();
int g = f();

// CHECK: define internal void @__cxx_global_var_init() [[ATTR:#[0-9]+]]
// CHECK: attributes [[ATTR]] = {{{.*}}"hlx-vector-stack-limit"="1024"

// NONE-NOT: "hlx-vector-stack-limit"

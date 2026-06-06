// RUN: %clang_cc1 -triple arm-unknown-linux -funwind-tables=1 -fsanitize-cfi-cross-dso -emit-llvm -o - %s | FileCheck %s

// CHECK: define weak {{.*}}void @__cfi_check({{.*}} [[ATTR:#[0-9]*]]

// CHECK: attributes [[ATTR]] = {{.*}} uwtable(sync)

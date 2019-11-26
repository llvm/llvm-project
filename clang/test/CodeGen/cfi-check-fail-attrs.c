// RUN: %clang_cc1 -triple aarch64-unknown-linux -fsanitize-cfi-cross-dso -target-feature +reserve-x18 -emit-llvm -o - %s | FileCheck %s

// CHECK: define weak_odr hidden void @__cfi_check_fail{{.*}} [[ATTR:#[0-9]*]]

// CHECK: attributes [[ATTR]] = {{.*}} "target-features"="+reserve-x18"

// RUN: %clang -target aarch64-arm-none-eabi -S -emit-llvm -o - -msign-return-address=none  %s | FileCheck %s --check-prefix=CHECK-NONE
// RUN: %clang -target aarch64-arm-none-eabi -S -emit-llvm -o - -msign-return-address=non-leaf %s | FileCheck %s --check-prefix=CHECK-PARTIAL
// RUN: %clang -target aarch64-arm-none-eabi -S -emit-llvm -o - -msign-return-address=all %s | FileCheck %s --check-prefix=CHECK-ALL

// CHECK-NONE: @foo() #[[ATTR:[0-9]*]]
// CHECK-NONE-NOT: attributes #[[ATTR]] = { {{.*}} "sign-return-address"={{.*}} }}

// CHECK-PARTIAL: @foo() #[[ATTR:[0-9]*]]
// CHECK-PARTIAL: attributes #[[ATTR]] = { {{.*}} "sign-return-address"="non-leaf" {{.*}}}

// CHECK-ALL: @foo() #[[ATTR:[0-9]*]]
// CHECK-ALL: attributes #[[ATTR]] = { {{.*}} "sign-return-address"="all" {{.*}} }

void foo() {}

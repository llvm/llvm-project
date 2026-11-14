// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -disable-llvm-passes \
// RUN:   -femit-all-decls -emit-llvm -o - %s | FileCheck %s

[[clang::oracle_function]] static long oracle(void) {
// CHECK: define {{.*}} void @oracle() [[ATTR1:#[0-9]+]]
}

void not_oracle() {
// CHECK: define {{.*}} void @not_oracle() [[ATTR2:#[0-9]+]]
}

// CHECK: attributes [[ATTR1]] = {{.*}} "oracle-function"
// CHECK: attributes [[ATTR2]] = {
// CHECK-NOT: "oracle-function"
// CHECK: }

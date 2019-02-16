// RUN: %clang -target aarch64-arm-none-eabi -S -emit-llvm -o - -msign-return-address=none  %s | \
// RUN: FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-NONE
// RUN: %clang -target aarch64-arm-none-eabi -S -emit-llvm -o - -msign-return-address=non-leaf %s | \
// RUN: FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-PARTIAL  --check-prefix=CHECK-A-KEY
// RUN: %clang -target aarch64-arm-none-eabi -S -emit-llvm -o - -msign-return-address=all %s | \
// RUN: FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-ALL  --check-prefix=CHECK-A-KEY

// RUN: %clang -target aarch64-arm-none-eabi -S -emit-llvm -o - -mbranch-protection=none %s | \
// RUN: FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-NONE
// RUN: %clang -target aarch64-arm-none-eabi -S -emit-llvm -o - -mbranch-protection=standard %s | \
// RUN: FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-PARTIAL  --check-prefix=CHECK-A-KEY  --check-prefix=CHECK-BTE
// RUN: %clang -target aarch64-arm-none-eabi -S -emit-llvm -o - -mbranch-protection=pac-ret %s | \
// RUN: FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-PARTIAL  --check-prefix=CHECK-A-KEY
// RUN: %clang -target aarch64-arm-none-eabi -S -emit-llvm -o - -mbranch-protection=pac-ret+leaf %s | \
// RUN: FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-ALL  --check-prefix=CHECK-A-KEY
// RUN: %clang -target aarch64-arm-none-eabi -S -emit-llvm -o - -mbranch-protection=pac-ret+b-key %s | \
// RUN: FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-PARTIAL  --check-prefix=CHECK-B-KEY
// RUN: %clang -target aarch64-arm-none-eabi -S -emit-llvm -o - -mbranch-protection=pac-ret+b-key+leaf %s | \
// RUN: FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-ALL  --check-prefix=CHECK-B-KEY
// RUN: %clang -target aarch64-arm-none-eabi -S -emit-llvm -o - -mbranch-protection=bti %s | \
// RUN: FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-BTE
// RUN: %clang -target aarch64-arm-none-eabi -S -emit-llvm -o - -mbranch-protection=pac-ret+b-key+leaf+bti %s | \
// RUN: FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-ALL  --check-prefix=CHECK-B-KEY --check-prefix=BTE

struct Foo {
  Foo() {}
  ~Foo() {}
};

Foo f;

// CHECK: @llvm.global_ctors {{.*}}i32 65535, void ()* @[[CTOR_FN:.*]], i8* null

// CHECK: @[[CTOR_FN]]() #[[ATTR:[0-9]*]]

// CHECK-NONE-NOT: "sign-return-address"={{.*}}
// CHECK-PARTIAL: "sign-return-address"="non-leaf"
// CHECK-ALL: "sign-return-address"="all"
// CHECK-A-KEY: "sign-return-address-key"="a_key"
// CHECK-B-KEY: "sign-return-address-key"="b_key"
// CHECK-BTE: "branch-target-enforcement"

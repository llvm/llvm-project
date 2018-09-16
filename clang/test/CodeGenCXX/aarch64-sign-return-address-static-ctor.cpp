// RUN: %clang -target aarch64-arm-none-eabi -S -emit-llvm -o - -msign-return-address=none  %s | \
// RUN: FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-NONE
// RUN: %clang -target aarch64-arm-none-eabi -S -emit-llvm -o - -msign-return-address=non-leaf %s | \
// RUN: FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-PARTIAL
// RUN: %clang -target aarch64-arm-none-eabi -S -emit-llvm -o - -msign-return-address=all %s | \
// RUN: FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-ALL

struct Foo {
  Foo() {}
  ~Foo() {}
};

Foo f;

// CHECK: @llvm.global_ctors {{.*}}i32 65535, void ()* @[[CTOR_FN:.*]], i8* null

// CHECK: @[[CTOR_FN]]() #[[ATTR:[0-9]*]]

// CHECK-NONE-NOT: attributes #[[ATTR]] = { {{.*}} "sign-return-address"={{.*}} }}
// CHECK-PARTIAL: attributes #[[ATTR]] = { {{.*}} "sign-return-address"="non-leaf" {{.*}}}
// CHECK-ALL: attributes #[[ATTR]] = { {{.*}} "sign-return-address"="all" {{.*}} }

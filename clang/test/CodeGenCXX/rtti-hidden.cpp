// RUN: %clang_cc1 %s -triple=x86_64-pc-linux -emit-llvm -o - | FileCheck %s
// RUN: %if cir-enabled %{ %clang_cc1 -fclangir %s -triple=x86_64-pc-linux -emit-llvm -o - | FileCheck %s %}

// Test that this is not hidden.
// CHECK: @_ZTVN10__cxxabiv120__si_class_type_infoE = external global

class foo {
  virtual void baz();
};
struct __attribute__((__visibility__("hidden"))) bar : public foo {};
bar zed;

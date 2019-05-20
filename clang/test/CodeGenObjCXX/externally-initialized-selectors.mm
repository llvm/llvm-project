// RUN: %clang_cc1 -fobjc-runtime=macosx-fragile-10.5 -o - -emit-llvm %s | FileCheck %s
// RUN: %clang_cc1 -o - -emit-llvm %s | FileCheck %s

// CHECK: @OBJC_SELECTOR_REFERENCES_ = private externally_initialized global

void test(id x) {
  [x doSomething];
}

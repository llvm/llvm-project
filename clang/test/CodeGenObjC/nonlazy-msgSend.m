// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm %s -o - | FileCheck %s

// CHECK: declare ptr @objc_msgSend(ptr, ptr, ...) [[NLB:#[0-9]+]]
void f0(id x) {
  [x foo];
}

// CHECK: attributes [[NLB]] = { nonlazybind }

// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm %s -o - | FileCheck %s

// CHECK: declare i8* @objc_msgSend(i8*, i8*, ...) [[NLB:#[0-9]+]]
void f0(id x) {
  [x foo];
}

// CHECK: attributes [[NLB]] = { nonlazybind }

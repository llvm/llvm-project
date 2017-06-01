// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -triple x86_64-apple-darwin -fobjc-runtime=macosx-10.10.0 -fblocks -fobjc-arc %s | FileCheck %s

@interface Foo
@end

// CHECK-LABEL: doSomething:
void doSomething() { // CHECK: File 0, [[@LINE]]:20 -> {{[0-9:]+}} = #0
  return;
  __block Foo *f; // CHECK: File 0, [[@LINE]]:3 -> {{[0-9:]+}} = 0
}

int main() {}

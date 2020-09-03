// RUN: %clang_cc1 -triple wasm32 -o - -emit-llvm %s | FileCheck %s

// Don't mangle the no-arg form of main.

int main(void) {
  return 0;
}

// CHECK-LABEL: define i32 @main()

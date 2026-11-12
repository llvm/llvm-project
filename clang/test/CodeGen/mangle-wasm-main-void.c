// RUN: %clang_cc1 -emit-llvm %s -o - -triple=wasm32-unknown-unknown | FileCheck %s

int main(void) {
  return 0;
}
// CHECK: @__main_void = hidden alias i32 (), ptr @main
// CHECK: define i32 @main() #0 {

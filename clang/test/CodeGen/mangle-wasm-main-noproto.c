// RUN: %clang_cc1 -emit-llvm %s -o - -triple=wasm32-unknown-unknown -Wno-deprecated-non-prototype | FileCheck %s

int main(argc, argv)
  int argc;
  char* argv[];
{
  return 0;
}
// CHECK-NOT: __main_void
// CHECK: define i32 @__main_argc_argv(i32 noundef %argc, ptr noundef %argv) #0 { 

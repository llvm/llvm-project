// RUN: %clang_cc1 -triple wasm32 -o - -emit-llvm %s | FileCheck %s

// Mangle the argc/argv form of main.

int main(int argc, char **argv) {
  return 0;
}

// CHECK-LABEL: define i32 @__main_argc_argv(i32 noundef %argc, i8** noundef %argv)

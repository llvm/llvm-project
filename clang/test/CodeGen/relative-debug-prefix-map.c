// RUN: mkdir -p %t.nested/dir && cd %t.nested/dir
// RUN: cp %s %t.nested/dir/main.c
// RUN: %clang_cc1 -debug-info-kind=standalone -fdebug-prefix-map=%t.nested=. %t.nested/dir/main.c -emit-llvm -o - | FileCheck %s
//
// RUN: %clang_cc1 -debug-info-kind=standalone -fdebug-prefix-map=%p=. %s -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-DIRECT
//
// RUN: cd %p
// RUN: %clang_cc1 -debug-info-kind=standalone -fdebug-compilation-dir=. relative-debug-prefix-map.c -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-DIRECT

// CHECK: !DIFile(filename: "main.c", directory: "./dir")
// CHECK-DIRECT: !DIFile(filename: "relative-debug-prefix-map.c", directory: ".")

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  return 0;
}

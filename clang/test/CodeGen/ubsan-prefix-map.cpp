// RUN: %clang %s -target x86_64-linux-gnu -emit-llvm -S -fsanitize=undefined -o - | FileCheck %s -check-prefix=REGULAR
// RUN: %clang %s -target x86_64-linux-gnu -emit-llvm -S -fsanitize=undefined -fsanitize-prefix-map=%/S/= -o - | FileCheck %s -check-prefix=REMAPPED
// Use %/S which normalizes path separators to forward slashes on all platforms, including Windows.

// REGULAR: @{{.*}} = {{.*}} c"{{.*test(.|\\\\)CodeGen(.|\\\\)ubsan-prefix-map\.cpp}}\00"
// REMAPPED: @{{.*}} = {{.*}} c"ubsan-prefix-map.cpp\00"

int f(int x, int y) {
  return x / y;
}

// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -emit-llvm -fsanitize=address -o - | FileCheck %s -check-prefix=REGULAR
// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -emit-llvm -fsanitize=address -fsanitize-prefix-map=%/S/= -o - | FileCheck %s -check-prefix=REMAPPED
// Use %/S which normalizes path separators to forward slashes on all platforms, including Windows.

// REGULAR: @___asan_gen_module = private constant [{{[0-9]+}} x i8] c"{{.*test(.|\\\\)CodeGen(.|\\\\)asan-prefix-map\.cpp}}\00"
// REMAPPED: @___asan_gen_module = private constant [{{[0-9]+}} x i8] c"asan-prefix-map.cpp\00"

int global;

void f() {
  global = 1;
}

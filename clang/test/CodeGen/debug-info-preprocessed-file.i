# 1 "/foo/bar/preprocessed-input.c"
# 1 "<built-in>" 1
# 1 "<built-in>" 3
# 318 "<built-in>" 3
# 1 "<command line>" 1
# 1 "<built-in>" 2
# 1 "preprocessed-input.c" 2

// RUN: %clang -g -c -S -emit-llvm -o - %s | FileCheck %s
// CHECK: !DICompileUnit(language: DW_LANG_C99, file: ![[FILE:[0-9]+]] 
// CHECK: ![[FILE]] = !DIFile(filename: "/foo/bar/preprocessed-input.c"

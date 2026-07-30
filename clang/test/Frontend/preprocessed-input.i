# 1 "preprocessed-input.c"
# 1 "<built-in>" 1
# 1 "<built-in>" 3
# 318 "<built-in>" 3
# 1 "<command line>" 1
# 1 "<built-in>" 2
# 1 "preprocessed-input.c" 2

// RUN: %clang -emit-llvm -S -o - %s | FileCheck %s
// CHECK: source_filename = "preprocessed-input.c"{{$}}

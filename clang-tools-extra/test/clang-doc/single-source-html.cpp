// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"
// RUN: clang-doc --format=html --executor=standalone %s -output=%t/docs | FileCheck %s

// CHECK: Using default asset: {{.*}}..\share\clang
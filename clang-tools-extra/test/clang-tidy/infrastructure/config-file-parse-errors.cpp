// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir
// RUN: echo 'not valid yaml: ][' > %t.dir/.clang-tidy
// RUN: cp %s %t.dir/test.cpp
// RUN: clang-tidy %t.dir/test.cpp -checks='misc-explicit-constructor' -- 2>&1 \
// RUN:   | FileCheck %s

// CHECK: Error parsing {{.*}}.clang-tidy:

class A { A(int) {} };

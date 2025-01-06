// Testing that we won't record the type ID from external modules.
//
// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/b.cppm -emit-module-interface -o %t/b.pcm \
// RUN:     -fmodule-file=a=%t/a.pcm
// RUN: llvm-bcanalyzer --dump --disable-histogram %t/b.pcm | FileCheck %t/b.cppm
//
// RUN: %clang_cc1 -std=c++20 %t/a.v1.cppm -emit-module-interface -o %t/a.v1.pcm
// RUN: %clang_cc1 -std=c++20 %t/b.cppm -emit-module-interface -o %t/b.v1.pcm \
// RUN:     -fmodule-file=a=%t/a.v1.pcm
// RUN: diff %t/b.pcm %t/b.v1.pcm &> /dev/null

//--- a.cppm
export module a;
export int a();

//--- b.cppm
export module b;
import a;
export int b();

// CHECK: <DECL_FUNCTION {{.*}} op8=4120
// CHECK: <TYPE_FUNCTION_PROTO

//--- a.v1.cppm
// We remove the unused the function and testing if the format of the BMI of B will change.
export module a;


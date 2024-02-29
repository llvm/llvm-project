// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/user.cpp -fmodule-file=%t/a.pcm -fsyntax-only \
// RUN:    2>&1 | FileCheck %t/user.cpp
// RUN: %clang_cc1 -std=c++20 %t/b.cppm -emit-module-interface -o %t/b.pcm \
// RUN:    -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 %t/b.pcm -S \
// RUN:    -fprebuilt-module-path=%t -emit-llvm 2>&1 -o - | FileCheck %t/b.cppm

//--- a.cppm
export module a;

//--- b.cppm
export module b;
import a;

// CHECK-NOT: warning

//--- user.cpp
import a;

// CHECK: the form '-fmodule-file=<BMI-path>' is deprecated for standard C++ named modules;consider to use '-fmodule-file=<module-name>=<BMI-path>' instead

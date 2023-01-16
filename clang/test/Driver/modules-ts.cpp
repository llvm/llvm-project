// Check compiling a module interface to a .pcm file.
//
// RUN: %clang -fmodules-ts -x c++-module --precompile %s -o %t.pcm -v 2>&1 | FileCheck %s --check-prefix=CHECK-PRECOMPILE
//
// CHECK-PRECOMPILE:  warning: the '-fmodules-ts' flag is deprecated and it will be removed in Clang 17; use '-std=c++20' or higher to use standard C++ modules instead [-Wdeprecated-module-ts]
// CHECK-PRECOMPILE: -cc1 {{.*}} -emit-module-interface
// CHECK-PRECOMPILE-SAME: -o {{.*}}.pcm
// CHECK-PRECOMPILE-SAME: -x c++
// CHECK-PRECOMPILE-SAME: modules-ts.cpp

// Check compiling a .pcm file to a .o file.
//
// RUN: %clang -fmodules-ts -fintegrated-as %t.pcm -c -o %t.pcm.o -v 2>&1 | FileCheck %s --check-prefix=CHECK-COMPILE --check-prefix=CHECK-WARN
//
// CHECK-WARN:  warning: the '-fmodules-ts' flag is deprecated and it will be removed in Clang 17; use '-std=c++20' or higher to use standard C++ modules instead [-Wdeprecated-module-ts]
// CHECK-COMPILE: -cc1 {{.*}} -emit-obj
// CHECK-COMPILE-SAME: -o {{.*}}.pcm.o
// CHECK-COMPILE-SAME: -x pcm
// CHECK-COMPILE-SAME: {{.*}}.pcm

// Check use of a .pcm file in another compilation.
//
// RUN: %clang -fmodules-ts -fmodule-file=%t.pcm -fintegrated-as -Dexport= %s -c -o %t.o -v 2>&1 | FileCheck %s --check-prefix=CHECK-USE
//
// CHECK-USE:  warning: the '-fmodules-ts' flag is deprecated and it will be removed in Clang 17; use '-std=c++20' or higher to use standard C++ modules instead [-Wdeprecated-module-ts]
// CHECK-USE: -cc1
// CHECK-USE-SAME: -emit-obj
// CHECK-USE-SAME: -fmodule-file={{.*}}.pcm
// CHECK-USE-SAME: -o {{.*}}.o{{"?}} {{.*}}-x c++
// CHECK-USE-SAME: modules-ts.cpp

// Check combining precompile and compile steps works.
//
// RUN: %clang -fmodules-ts -fintegrated-as -x c++-module %s -c -o %t.pcm.o -v 2>&1 | FileCheck %s --check-prefix=CHECK-PRECOMPILE --check-prefix=CHECK-COMPILE

// Check that .cppm is treated as a module implicitly.
// RUN: cp %s %t.cppm
// RUN: %clang -fmodules-ts --precompile %t.cppm -o %t.pcm -v 2>&1 | FileCheck %s --check-prefix=CHECK-PRECOMPILE

// Note, we use -Dexport= to make this a module implementation unit when building the implementation.
export module foo;

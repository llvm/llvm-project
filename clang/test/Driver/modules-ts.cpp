// Check compiling a module interface to a .pcm file.
//
// RUN: %clang -fmodules-ts -x c++-module --precompile %s -Dimplementation= -o %t.pcm -v 2>&1 | FileCheck %s --check-prefix=CHECK-PRECOMPILE
//
// CHECK-PRECOMPILE: -cc1 {{.*}} -emit-module-interface
// CHECK-PRECOMPILE-SAME: -o {{.*}}.pcm
// CHECK-PRECOMPILE-SAME: -x c++-module
// CHECK-PRECOMPILE-SAME: modules-ts.cpp

// Check compiling a .pcm file to a .o file.
//
// RUN: %clang -fmodules-ts %t.pcm -c -o %t.pcm.o -v 2>&1 | FileCheck %s --check-prefix=CHECK-COMPILE
//
// CHECK-COMPILE: -cc1 {{.*}} -emit-obj
// CHECK-COMPILE-SAME: -o {{.*}}.pcm.o
// CHECK-COMPILE-SAME: -x pcm
// CHECK-COMPILE-SAME: {{.*}}.pcm

// Check use of a .pcm file in another compilation.
//
// RUN: %clang -fmodules-ts -fmodule-file=%t.pcm %s -c -o %t.o -v 2>&1 | FileCheck %s --check-prefix=CHECK-USE
//
// CHECK-USE: -cc1
// CHECK-USE-SAME: -emit-obj
// CHECK-USE-SAME: -fmodule-file={{.*}}.pcm
// CHECK-USE-SAME: -o {{.*}}.o{{"?}} {{.*}}-x c++
// CHECK-USE-SAME: modules-ts.cpp

// Check combining precompile and compile steps works.
//
// RUN: %clang -fmodules-ts -x c++-module %s -Dimplementation= -c -o %t.pcm.o -v 2>&1 | FileCheck %s --check-prefix=CHECK-PRECOMPILE --check-prefix=CHECK-COMPILE

// Check that .cppm is treated as a module implicitly.
// RUN: cp %s %t.cppm
// RUN: %clang -fmodules-ts --precompile %t.cppm -Dimplementation= -o %t.pcm -v 2>&1 | FileCheck %s --check-prefix=CHECK-PRECOMPILE

// Note, we use -Dimplementation= to make this a valid module interface unit when building the interface.
module implementation foo;

// RUN: rm -rf %t
// RUN: mkdir %t

// Check compiling a module interface to a .pcm file.
//
// RUN: %clang -std=c++2a -x c++-module --precompile %s -o %t/module.pcm -v 2>&1 | FileCheck %s --check-prefix=CHECK-PRECOMPILE
// RUN: %clang -std=gnu++2a -x c++-module --precompile %s -o %t/module-gnu.pcm -v 2>&1 | FileCheck %s --check-prefix=CHECK-PRECOMPILE
//
// CHECK-PRECOMPILE: -cc1 {{.*}} -emit-module-interface
// CHECK-PRECOMPILE-SAME: -o {{.*}}.pcm
// CHECK-PRECOMPILE-SAME: -x c++
// CHECK-PRECOMPILE-SAME: modules.cpp

// Check compiling a .pcm file to a .o file.
//
// RUN: %clang -std=c++2a %t/module.pcm -S -o %t/module.pcm.o -v 2>&1 | FileCheck %s --check-prefix=CHECK-COMPILE
//
// CHECK-COMPILE: -cc1 {{.*}} {{-emit-obj|-S}}
// CHECK-COMPILE-SAME: -o {{.*}}module{{2*}}.pcm.o
// CHECK-COMPILE-SAME: -x pcm
// CHECK-COMPILE-SAME: {{.*}}.pcm

// Check use of a .pcm file in another compilation.
//
// RUN: %clang -std=c++2a -fmodule-file=%t/module.pcm -Dexport= %s -S -o %t/module.o -v 2>&1 | FileCheck %s --check-prefix=CHECK-USE
// RUN: %clang -std=c++20 -fmodule-file=%t/module.pcm -Dexport= %s -S -o %t/module.o -v 2>&1 | FileCheck %s --check-prefix=CHECK-USE
// RUN: %clang -std=gnu++20 -fmodule-file=%t/module-gnu.pcm -Dexport= %s -S -o %t/module.o -v 2>&1 | FileCheck %s --check-prefix=CHECK-USE
//
// CHECK-USE: -cc1 {{.*}} {{-emit-obj|-S}}
// CHECK-USE-SAME: -fmodule-file={{.*}}.pcm
// CHECK-USE-SAME: -o {{.*}}.{{o|s}}{{"?}} {{.*}}-x c++
// CHECK-USE-SAME: modules.cpp

// Check combining precompile and compile steps works.
//
// RUN: %clang -std=c++2a -x c++-module %s -S -o %t/module2.pcm.o -v 2>&1 | FileCheck %s --check-prefix=CHECK-PRECOMPILE --check-prefix=CHECK-COMPILE

// Check that .cppm is treated as a module implicitly.
//
// RUN: cp %s %t/module.cppm
// RUN: %clang -std=c++2a --precompile %t/module.cppm -o %t/module.pcm -v 2>&1 | FileCheck %s --check-prefix=CHECK-PRECOMPILE

export module foo;

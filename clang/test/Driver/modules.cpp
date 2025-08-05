// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t

// Check compiling a module interface to a .pcm file.
//
// RUN: %clang -std=c++2a -x c++-module --precompile %t/foo.cpp -o %t/foo.pcm -v 2>&1 | FileCheck %s --check-prefix=CHECK-PRECOMPILE
// RUN: %clang -std=gnu++2a -x c++-module --precompile %t/foo.cpp -o %t/foo-gnu.pcm -v 2>&1 | FileCheck %s --check-prefix=CHECK-PRECOMPILE
//
// CHECK-PRECOMPILE: -cc1 {{.*}} -emit-module-interface
// CHECK-PRECOMPILE-SAME: -o {{.*}}.pcm
// CHECK-PRECOMPILE-SAME: -x c++
// CHECK-PRECOMPILE-SAME: foo.cpp

// Check compiling a .pcm file to a .o file.
//
// RUN: %clang -std=c++2a %t/foo.pcm -S -o %t/foo.pcm.o -v 2>&1 | FileCheck %s --check-prefix=CHECK-COMPILE
//
// CHECK-COMPILE: -cc1 {{.*}} {{-emit-obj|-S}}
// CHECK-COMPILE-SAME: -o {{.*}}foo{{2*}}.pcm.o
// CHECK-COMPILE-SAME: -x pcm
// CHECK-COMPILE-SAME: {{.*}}.pcm

// Check use of a .pcm file in another compilation.
//
// RUN: %clang -std=c++2a -fmodule-file=foo=%t/foo.pcm %t/foo_impl.cpp -S -o %t/module.o -v 2>&1 | FileCheck %s --check-prefix=CHECK-USE
// RUN: %clang -std=c++20 -fmodule-file=foo=%t/foo.pcm %t/foo_impl.cpp -S -o %t/module.o -v 2>&1 | FileCheck %s --check-prefix=CHECK-USE
// RUN: %clang -std=gnu++20 -fmodule-file=foo=%t/foo-gnu.pcm %t/foo_impl.cpp -S -o %t/module.o -v 2>&1 | FileCheck %s --check-prefix=CHECK-USE
//
// CHECK-USE: -cc1 {{.*}} {{-emit-obj|-S}}
// CHECK-USE-SAME: -fmodule-file=foo={{.*}}.pcm
// CHECK-USE-SAME: -o {{.*}}.{{o|s}}{{"?}} {{.*}}-x c++
// CHECK-USE-SAME: foo_impl.cpp

// Check combining precompile and compile steps works.
//
// RUN: %clang -std=c++2a -x c++-module -fno-modules-reduced-bmi %t/foo.cpp -S -o %t/foo2.pcm.o -v 2>&1 | FileCheck %s --check-prefix=CHECK-PRECOMPILE --check-prefix=CHECK-COMPILE

// Check that .cppm is treated as a module implicitly.
//
// RUN: cp %t/foo.cpp %t/foo.cppm
// RUN: %clang -std=c++2a --precompile %t/foo.cppm -o %t/foo.pcm -v 2>&1 | FileCheck %s --check-prefix=CHECK-PRECOMPILE

//--- foo.cpp
export module foo;

//--- foo_impl.cpp
module foo;

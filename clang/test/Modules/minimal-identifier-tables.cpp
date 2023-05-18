// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo 'extern int some_long_variable_name;' > %t/x.h
// RUN: echo 'extern int some_long_variable_name;' > %t/y.h
// RUN: echo 'module X { header "x.h" } module Y { header "y.h" }' > %t/map
// RUN: %clang_cc1 -fmodules -x c++ -fmodule-name=X %t/map -emit-module -o %t/x.pcm
// RUN: %clang_cc1 -fmodules -x c++ -fmodule-name=Y %t/map -fmodule-file=%t/x.pcm -emit-module -o %t/y.pcm
// RUN: cat %t/y.pcm | FileCheck %s
//
// CHECK-NOT: some_long_variable_name

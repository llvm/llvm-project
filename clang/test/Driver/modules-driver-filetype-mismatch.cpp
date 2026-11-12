// Checks that only '-x c++module' inputs can be part of a module definition.
//
// RUN: not %clang -### -fmodules-driver -std=c++20 -x c++ %s 2>&1 \
// RUN:   | sed 's:\\\\\?:/:g' \
// RUN:   | FileCheck -DTHISFILE=%/s %s
//
// CHECK: clang: error: module 'M' is defined in file '[[THISFILE]]', but module declarations are only allowed in C++ module inputs; use the '.cppm' extension or '-x c++module'
export module M;

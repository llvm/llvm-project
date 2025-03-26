// REQUIRES: asserts

// RUN: %clang_cc1 -std=c++23 -x c++-header -emit-pch -fmodule-format=obj \
// RUN:   -o %t.pch %s \
// RUN:   -mllvm -debug-only=pchcontainer &>%t-pch.ll
// RUN: cat %t-pch.ll | FileCheck %s

template<class...>                     
using __void_t [[gnu::nodebug]] = void;
                                       
__void_t<> func() {}                   

// CHECK: !DICompileUnit
// CHECK-NOT: __void_t

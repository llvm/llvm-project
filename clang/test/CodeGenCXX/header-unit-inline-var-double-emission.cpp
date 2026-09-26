// RUN: rm -rf %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple -xc++-user-header -emit-header-unit %t/a.h -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple -xc++-user-header -emit-header-unit -fmodule-file=%t/a.pcm %t/b.h -o %t/b.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple -fmodule-file=%t/a.pcm -fmodule-file=%t/b.pcm %t/tu.cpp \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %t/tu.cpp

//--- a.h
struct P {
  constexpr P() : x(0) {}
  ~P();
  union {
    int x;
    long y;
  };
};
inline P S;

//--- b.h
import "a.h";
inline const int *PTR = &S.x;

//--- tu.cpp
import "a.h";
import "b.h";

// CHECK: @S = linkonce_odr{{.*}} global { { i32, [4 x i8] } }
// CHECK: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @__cxx_global_var_init, ptr @S }]
// CHECK: define internal void @__cxx_global_var_init()
// CHECK-NOT: define {{.*}} @__cxx_global_var_init

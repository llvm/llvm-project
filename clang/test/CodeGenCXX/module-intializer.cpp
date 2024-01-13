// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t

// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 N.cpp \
// RUN:    -emit-module-interface -o N.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 N.pcm -S -emit-llvm \
// RUN:  -o - | FileCheck %s --check-prefix=CHECK-N

// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 O.cpp \
// RUN:    -emit-module-interface -o O.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 O.pcm -S -emit-llvm \
// RUN:  -o - | FileCheck %s --check-prefix=CHECK-O

// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 M-Part.cpp \
// RUN:    -emit-module-interface -o M-Part.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 M-Part.pcm -S \
// RUN:    -emit-module-interface  -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-P

// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 M.cpp \
// RUN:    -fprebuilt-module-path=%t -emit-module-interface -o M.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 M.pcm -S -emit-llvm \
// RUN:    -fprebuilt-module-path=%t -o - | FileCheck %s --check-prefix=CHECK-M

// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 useM.cpp \
// RUN:   -fprebuilt-module-path=%t -S -emit-llvm  -o - \
// RUN:   | FileCheck %s --check-prefix=CHECK-USE

// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 M-impl.cpp \
// RUN:   -fprebuilt-module-path=%t -S -emit-llvm  -o - \
// RUN:   | FileCheck %s --check-prefix=CHECK-IMPL

// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 N.cpp -S -emit-llvm \
// RUN:   -o - | FileCheck %s --check-prefix=CHECK-N

// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 O.cpp -S -emit-llvm \
// RUN:   -o - | FileCheck %s --check-prefix=CHECK-O

// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 M-Part.cpp -S -emit-llvm \
// RUN:   -o - | FileCheck %s --check-prefix=CHECK-P

// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 M.cpp \
// RUN:   -fprebuilt-module-path=%t -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-M

//--- N-h.h

struct Oink {
  Oink(){};
};

Oink Hog;

//--- N.cpp

module;
#include "N-h.h"

export module N;

export struct Quack {
  Quack(){};
};

export Quack Duck;

// CHECK-N: define internal void @__cxx_global_var_init
// CHECK-N: call {{.*}} @_ZN4OinkC1Ev
// CHECK-N: define internal void @__cxx_global_var_init
// CHECK-N: call {{.*}} @_ZNW1N5QuackC1Ev
// CHECK-N: define void @_ZGIW1N
// CHECK-N: store i8 1, ptr @_ZGIW1N__in_chrg
// CHECK-N: call void @__cxx_global_var_init
// CHECK-N: call void @__cxx_global_var_init

//--- O-h.h

struct Meow {
  Meow(){};
};

Meow Cat;

//--- O.cpp

module;
#include "O-h.h"

export module O;

export struct Bark {
  Bark(){};
};

export Bark Dog;

// CHECK-O: define internal void @__cxx_global_var_init
// CHECK-O: call {{.*}} @_ZN4MeowC2Ev
// CHECK-O: define internal void @__cxx_global_var_init
// CHECK-O: call {{.*}} @_ZNW1O4BarkC1Ev
// CHECK-O: define void @_ZGIW1O
// CHECK-O: store i8 1, ptr @_ZGIW1O__in_chrg
// CHECK-O: call void @__cxx_global_var_init
// CHECK-O: call void @__cxx_global_var_init

//--- P-h.h

struct Croak {
  Croak(){};
};

Croak Frog;

//--- M-Part.cpp

module;
#include "P-h.h"

module M:Part;

struct Squawk {
  Squawk(){};
};

Squawk parrot;

// CHECK-P: define internal void @__cxx_global_var_init
// CHECK-P: call {{.*}} @_ZN5CroakC1Ev
// CHECK-P: define internal void @__cxx_global_var_init
// CHECK-P: call {{.*}} @_ZNW1M6SquawkC1Ev
// CHECK-P: define void @_ZGIW1MWP4Part
// CHECK-P: store i8 1, ptr @_ZGIW1MWP4Part__in_chrg
// CHECK-P: call void @__cxx_global_var_init
// CHECK-P: call void @__cxx_global_var_init

//--- M-h.h

struct Moo {
  Moo(){};
};

Moo Cow;

//--- M.cpp

module;
#include "M-h.h"

export module M;
import N;
export import O;
import :Part;

export struct Baa {
  int x;
  Baa(){};
  Baa(int x) : x(x) {}
  int getX() { return x; }
};

export Baa Sheep(10);

// CHECK-M: define internal void @__cxx_global_var_init
// CHECK-M: call {{.*}} @_ZN3MooC1Ev
// CHECK-M: define internal void @__cxx_global_var_init
// CHECK-M: call {{.*}} @_ZNW1M3BaaC1Ei
// CHECK-M: declare void @_ZGIW1O()
// CHECK-M: declare void @_ZGIW1N()
// CHECK-M: declare void @_ZGIW1MWP4Part()
// CHECK-M: define void @_ZGIW1M
// CHECK-M: store i8 1, ptr @_ZGIW1M__in_chrg
// CHECK-M: call void @_ZGIW1O()
// CHECK-M: call void @_ZGIW1N()
// CHECK-M: call void @_ZGIW1MWP4Part()
// CHECK-M: call void @__cxx_global_var_init
// CHECK-M: call void @__cxx_global_var_init

//--- useM.cpp

import M;

int main() {
  return Sheep.getX();
}

// CHECK-USE: declare void @_ZGIW1M
// CHECK-USE: define internal void @_GLOBAL__sub_I_useM.cpp
// CHECK-USE: call void @_ZGIW1M()

//--- M-impl.cpp

module M;

int foo(int i) { return i + 1; }

// CHECK-IMPL: declare void @_ZGIW1M
// CHECK-IMPL: define internal void @_GLOBAL__sub_I_M_impl.cpp
// CHECK-IMPL: call void @_ZGIW1M()

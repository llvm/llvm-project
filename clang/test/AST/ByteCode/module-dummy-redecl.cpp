// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -fmodules -fmodules-cache-path=%t -fmodule-map-file=%t/module.modulemap %t/use.cpp -emit-llvm -o - -triple x86_64-linux-gnu
// RUN: %clang_cc1 -std=c++20 -fmodules -fmodules-cache-path=%t -fmodule-map-file=%t/module.modulemap %t/use.cpp -emit-llvm -o - -triple x86_64-linux-gnu -fexperimental-new-constant-interpreter

//--- module.modulemap
module b { header "b.h" export * }
module c { header "c.h" export * }

//--- nonmodular.h
template<typename T> int Z;

//--- b.h
#include "nonmodular.h"

inline constexpr int *xP = &Z<decltype([] { static int n; return &n; }())>;

//--- c.h
#include "nonmodular.h"

inline constexpr int *yP = &Z<decltype([] { static int n; return &n; }())>;


//--- use.cpp
#include "nonmodular.h"

inline constexpr int *P = &Z<decltype([] { static int n; return &n; }())>;

#include "b.h"
#include "c.h"
static_assert(xP == yP);

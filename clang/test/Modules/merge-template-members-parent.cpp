// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -emit-obj -fmodules -fimplicit-module-maps -fmodules-cache-path=%t %t/merge.cpp -o %t/merge.o

//--- V.h
#ifndef V_H
#define V_H
template <typename T>
struct V {
  ~V() {}
};
#endif

//--- A.h
#include "V.h"

void A(const V<unsigned long> &v);

//--- B.h
#include "V.h"

inline V<unsigned long> B() {
  return {};
}

//--- C.h
#include "V.h"

#include "A.h"

class C {
public:
  C(const V<unsigned long> &v) {
    V<unsigned long> v2;
  }
};

C GetC() {
   return {{}};
}

// This include *MUST* come last.
#include "B.h"

//--- module.modulemap
module "V" { header "V.h" export * }
module "A" { header "A.h" export * }
module "B" { header "B.h" export * }
module "C" { header "C.h" export * }

//--- merge.cpp
#include "C.h"

template <typename T>
C GetC_main() {
   return {{}};
}

void f() {
   GetC_main<float>();
   GetC();
}

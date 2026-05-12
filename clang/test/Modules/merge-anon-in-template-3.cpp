// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -fmodule-name=hu-01 -emit-header-unit -xc++-user-header %t/hu-01.h \
// RUN:  -o %t/hu-01.pcm

// RUN: %clang_cc1 -std=c++20 -fmodule-name=hu-02 -emit-header-unit -xc++-user-header %t/hu-02.h \
// RUN:  -Wno-experimental-header-units \
// RUN:  -fmodule-map-file=%t/hu-01.map -fmodule-file=hu-01=%t/hu-01.pcm \
// RUN:  -o %t/hu-02.pcm

// RUN: %clang_cc1 -std=c++20 -emit-obj %t/main.cpp \
// RUN:  -Wno-experimental-header-units \
// RUN:  -fmodule-map-file=%t/hu-01.map -fmodule-file=hu-01=%t/hu-01.pcm \
// RUN:  -fmodule-map-file=%t/hu-02.map -fmodule-file=hu-02=%t/hu-02.pcm

//--- hu-01.map
module "hu-01" {
  header "hu-01.h"
  export *
}

//--- hu-02.map
module "hu-02" {
  header "hu-02.h"
  export *
}

//--- hu-01.h
template <typename T>
struct S { union { T x; }; };

using SI = S<int>;

//--- hu-02.h
import "hu-01.h";
inline void f(S<int> s = {}) { s.x; }

//--- main.cpp
import "hu-01.h";
void g(S<int>) {}

import "hu-02.h";
void h() { f(); }

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: cd %t

// RUN: %clang_cc1 -fmodule-name=stl -fno-cxx-modules -emit-module -fmodules -xc++ stl.cppmap -o stl.pcm
// RUN: %clang_cc1 -fmodule-name=d -fno-cxx-modules -emit-module -fmodules -fmodule-file=stl.pcm -xc++ d.cppmap -o d.pcm
// RUN: %clang_cc1 -fmodule-name=b -fno-cxx-modules -emit-module -fmodules -fmodule-file=stl.pcm -xc++ b.cppmap -o b.pcm
// RUN: %clang_cc1 -fmodule-name=a -fno-cxx-modules -emit-module -fmodules -fmodule-file=stl.pcm -fmodule-file=d.pcm -fmodule-file=b.pcm -xc++ a.cppmap -o a.pcm
// RUN: %clang_cc1 -fno-cxx-modules -fmodules -fmodule-file=a.pcm -emit-llvm -o /dev/null main.cpp

//--- a.cppmap
module "a" {
header "a.h"
}

//--- a.h
#include "b.h"
namespace {
void a(absl::set<char> c) {
  absl::set<int> b;
  c.end();
  c.contains();
}
}  // namespace

//--- b.cppmap
module "b" {
header "b.h"
}

//--- b.h
#include "c.h"
void b() { absl::set<char> x; }

//--- c.h
#include "stl.h"
namespace absl {
template <typename>
class set {
 public:
  struct iterator {
    void u() const;
  };
  iterator end() const { return {}; }
  void contains() const { end().u(); }
  pair<iterator> e();
};
}  // namespace absl

//--- d.cppmap
module "d" {
header "d.h"
}

//--- d.h
#include "c.h"
void d() { absl::set<char> x; }

//--- stl.cppmap
module "stl" {
header "stl.h"
}

//--- stl.h
#ifndef _STL_H_
#define _STL_H_
template <class>
struct pair;
#endif

//--- main.cpp
// expected-no-diagnostics
#include "c.h"
void f(absl::set<char> o) { o.contains(); }

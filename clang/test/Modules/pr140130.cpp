// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: cd %t
// RUN: %clang_cc1 -iquote . -fmodules -fno-cxx-modules -emit-module \
// RUN:   -std=c++20 -fmodule-name=c -xc++ c.cppmap -o c.pcm
// RUN: %clang_cc1 -iquote . -fmodules -fno-cxx-modules -emit-module \
// RUN:   -std=c++20 -fmodule-name=a -fmodule-map-file=a.cppmap \
// RUN:   -fmodule-file=c.pcm -xc++ a.cppmap -o a.pcm

//--- a.cppmap
module "a" {
 header "a.h"
}
//--- a.h
#include "b.h"
//--- b.h
#ifndef _B_H_
#define _B_H_
struct B {
  consteval B() {}
  union {
    int a;
  };
};
constexpr B b;
#endif
//--- c.cppmap
module "c" {
header "c.h"
}
//--- c.h
#include "b.h"

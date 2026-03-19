// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -fallow-pcm-with-compiler-errors -fmodule-name=c  -xc++ -emit-module -fmodules -std=gnu++20 %t/a.cppmap -o %t/c.pcm
//--- a.cppmap
module "c" {
 header "a.h"
}

//--- a.h
template <class>
class C {};
template <class T>
C<T>::operator C() {}

// UNSUPPORTED: system-darwin
// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: mkdir %t/incl
// RUN: mv %t/header.h %t/incl/header.h
// RUN: cd %t
// RUN: %clang %target_itanium_abi_host_triple -g -o %t/a.out \
// RUN:   -Xclang -gsimple-template-names=mangled \
// RUN:   -Xclang -debug-forward-template-params \
// RUN:   -std=c++20 -fdebug-types-section -I incl a.cpp b.cpp
// RUN: llvm-dwarfdump --verify %t/a.out

//--- header.h
template <typename T> struct t1 {};
inline auto f1() {
  auto T = [] {};
  t1<decltype(T)> v;
  return v;
}
inline auto f2() {
  struct {
  } T;
  t1<decltype(T)> v;
  return v;
}
void a();
//--- a.cpp
#include "incl/header.h"
template <typename T> void ft() {}
void a() {
  ft<decltype(f1())>();
  ft<decltype(f2())>();
}
//--- b.cpp
#include "header.h"
template <typename T> void ft() {}
int main() {
  a();
  ft<decltype(f1())>();
  ft<decltype(f2())>();
}

//===-- ns.h ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <cstdio>

void test_lookup_at_global_scope();
void test_lookup_at_file_scope();
void test_lookup_before_using_directive();
void test_lookup_after_using_directive();
int func(int a);
namespace A {
int foo();
int func(int a);
inline int func() {
  std::printf("A::func()\n");
  return 3;
}
inline int func2() {
  std::printf("A::func2()\n");
  return 3;
}
void test_lookup_at_ns_scope();
namespace B {
int func();
void test_lookup_at_nested_ns_scope();
void test_lookup_at_nested_ns_scope_after_using();
} // namespace B
} // namespace A

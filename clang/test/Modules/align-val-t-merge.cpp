// Tests that an implicitly-declared std::align_val_t (created by
// DeclareGlobalNewDelete when a virtual destructor is seen) merges correctly
// with the module's explicit definition of std::align_val_t.
//
// Without the fix, the implicit align_val_t was not added to the std namespace
// DeclContext, so the ASTReader's noload_lookup couldn't find it during module
// deserialization. This resulted in two unmerged align_val_t types and a
// "no matching function for call to '__builtin_operator_delete'" error when
// module code passed the module's align_val_t to the implicitly declared
// operator delete (which uses the implicit align_val_t).
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// Build the module that provides std::align_val_t and a template using
// __builtin_operator_delete. Crucially, the module does NOT declare
// operator delete -- only the implicit declarations exist.
// RUN: %clang_cc1 -std=c++17 -x c++ -fmodules -fno-implicit-modules \
// RUN:   -faligned-allocation -fsized-deallocation \
// RUN:   -emit-module -fmodule-name=alloc \
// RUN:   -fmodule-map-file=%t/alloc.modulemap \
// RUN:   %t/alloc.modulemap -o %t/alloc.pcm
//
// Compile a TU that has a virtual destructor (triggers DeclareGlobalNewDelete
// and implicit align_val_t) BEFORE importing the module.
// RUN: %clang_cc1 -std=c++17 -x c++ -fmodules -fno-implicit-modules \
// RUN:   -faligned-allocation -fsized-deallocation \
// RUN:   -fmodule-map-file=%t/alloc.modulemap \
// RUN:   -fmodule-file=alloc=%t/alloc.pcm \
// RUN:   -fsyntax-only -verify %t/use.cpp

//--- alloc.modulemap
module alloc {
  header "alloc.h"
}

//--- alloc.h
#ifndef ALLOC_H
#define ALLOC_H

namespace std {
  using size_t = decltype(sizeof(0));
  enum class align_val_t : size_t {};
}

// No explicit operator delete declarations here -- we rely on the implicit
// ones from DeclareGlobalNewDelete. This mirrors the libc++ setup after
// the removal of the global_new_delete.h include from allocate.h.

template <class T>
void dealloc(T *p) {
  // __builtin_operator_delete resolves against the usual (implicit)
  // deallocation functions. Those use the implicit align_val_t.
  // The argument here uses the module's align_val_t.
  // If they're not merged, this fails with a type mismatch.
  __builtin_operator_delete(p, std::align_val_t(alignof(T)));
}

#endif

//--- use.cpp
// expected-no-diagnostics

// Virtual destructor triggers DeclareGlobalNewDelete(), which implicitly
// creates std::align_val_t before the module is loaded.
class Foo {
  virtual ~Foo() {}
};

#include "alloc.h"

void test() {
  dealloc<Foo>(nullptr);
}

// Verify that an internal-linkage function declared in the global module
// fragment of the current translation unit remains usable from within the
// module unit when it is only reached through a template instantiation that is
// performed after the global module fragment is closed (i.e. the instantiation
// is triggered by another entity in the global module fragment).
//
// This is a regression test: such a plain 'static' (non-inline) helper used to
// be wrongly removed from the overload set, producing a bogus "no matching
// function" error with no candidate notes.
//
// RUN: %clang_cc1 -std=c++20 %s -emit-module-interface -o %t.pcm -verify
// RUN: %clang_cc1 -std=c++23 %s -emit-module-interface -o %t.pcm -verify

// expected-no-diagnostics

module;
static void slow(unsigned long *o) { *o = 0; }
static void slow(unsigned int *o) { *o = 0; }

template <typename T> void parse(T *o) { slow(o); }

// These inline functions are in the global module fragment; the template
// specializations they use are instantiated when the fragment is closed.
inline unsigned long read64() {
  unsigned long t;
  parse(&t);
  return t;
}
inline unsigned int read32() {
  unsigned int t;
  parse(&t);
  return t;
}

export module a;

export inline unsigned long use() { return read64() + read32(); }

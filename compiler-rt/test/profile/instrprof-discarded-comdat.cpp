// Check that instrprof does not introduce references to discarded sections when
// using comdats.
//
// Occasionally, it is possible that the same function can be compiled in
// different TUs with slightly different linkages, e.g., due to different
// compiler options. However, if these are comdat functions, a single
// implementation will be chosen at link time. we want to ensure that the
// profiling data does not contain a reference to the discarded section.

// UNSUPPORTED: target={{.*windows.*}}

// RUN: mkdir -p %t.d
// RUN: %clangxx_pgogen -O2 -fPIC -ffunction-sections -fdata-sections -c %s -o %t.d/a1.o -DOBJECT_1 -mllvm -disable-preinline
// RUN: %clangxx_pgogen -O2 -fPIC -ffunction-sections -fdata-sections -c %s -o %t.d/a2.o
// RUN: %clangxx_pgogen -fPIC -shared -o %t.d/liba.so %t.d/a1.o %t.d/a2.o 2>&1 | FileCheck %s --allow-empty

// Ensure that we don't get an error when linking
// CHECK-NOT: relocation refers to a discarded section: .text._ZN1CIiE1fEi

template <typename T> struct C {
  void f(T x);
  int g(T x) {
    f(x);
    return v;
  }
  int v;
};

template <typename T>
#ifdef OBJECT_1
__attribute__((weak))
#else
__attribute__((noinline))
#endif
void C<T>::f(T x) {
  v += x;
}

#ifdef OBJECT_1
int foo() {
  C<int> c;
  c.f(1);
  return c.g(2);
}
#else
int bar() {
  C<int> c;
  c.f(3);
  return c.g(4);
}
#endif

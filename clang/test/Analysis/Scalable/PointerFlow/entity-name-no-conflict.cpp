// REQUIRES: asserts
// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fsyntax-only -std=c++20 %s \
// RUN:   --ssaf-extract-summaries=PointerFlow \
// RUN:   --ssaf-tu-summary-file=%t/tu.summary.json \
// RUN:   --ssaf-compilation-unit-id="tu-1" \
// RUN:   -mllvm -debug-only=ssaf-analyses 2>&1 | FileCheck %s --allow-empty


// The two `Holder<decltype([]{})>` instantiations are distinct types
// (each lambda is its own closure record), but the USR generator
// currently fails to distinguish them.


// CHECK-NOT: dropping duplicate PointerFlow summary

template <class T>
struct Holder {
  T *p;
  void reset(T *x) { p = x; }
};

void caller(int x) {
  Holder<decltype([]{})>().reset(nullptr);
  Holder<decltype([]{})>().reset(nullptr);
}

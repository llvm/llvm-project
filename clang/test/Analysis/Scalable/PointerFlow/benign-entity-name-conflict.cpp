// REQUIRES: asserts
// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fsyntax-only -std=c++20 %s \
// RUN:   --ssaf-extract-summaries=PointerFlow \
// RUN:   --ssaf-tu-summary-file=%t/tu.summary.json \
// RUN:   --ssaf-compilation-unit-id="tu-1" \
// RUN:   -mllvm -debug-only=ssaf-analyses 2>&1 | FileCheck %s

// The two `Holder<decltype([]{})>` instantiations have distinct types but
// produce colliding USRs for `reset`. The extractor must keep one summary and
// drop the other with a diagnostic, instead of crashing.

// CHECK: dropping duplicate PointerFlow summary

template <class T>
struct Holder {
  T *p;
  void reset(T *x) { p = x; }
};

void caller(int x) {
  Holder<decltype([]{})>().reset(nullptr);
  Holder<decltype([]{})>().reset(nullptr);
}

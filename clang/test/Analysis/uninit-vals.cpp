// RUN: %clang_analyze_cc1 -analyzer-checker=core.builtin -verify -DCHECK_FOR_CRASH %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify -analyzer-output=text %s

#ifdef CHECK_FOR_CRASH
// expected-no-diagnostics
#endif

namespace PerformTrivialCopyForUndefs {
struct A {
  int x;
};

struct B {
  A a;
};

struct C {
  B b;
};

void foo() {
  C c1;
  C *c2;
#ifdef CHECK_FOR_CRASH
  // If the value of variable is not defined and checkers that check undefined
  // values are not enabled, performTrivialCopy should be able to handle the
  // case with undefined values, too.
  c1.b.a = c2->b.a;
#else
  c1.b.a = c2->b.a; // expected-warning{{1st function call argument is an uninitialized value}}
                    // expected-note@-1{{1st function call argument is an uninitialized value}}
#endif
}
}

namespace gh_178797 {
struct SpecialBuffer {
    SpecialBuffer() : src(defaultBuffer), dst(defaultBuffer) {}
    int* src;
    int* dst;
    int defaultBuffer[2];
};
// Not really a swap, but we need an assignment assigning UndefinedVal
// within a "swap" function to trigger this behavior.
void swap(int& lhs, int& rhs) {
    lhs = rhs; // no-crash
    // Not reporting copying uninitialized data because that is explicitly suppressed in the checker.
}
void entry_point() {
    SpecialBuffer special;
    swap(*special.dst, *++special.src);
}
}  // namespace gh_178797

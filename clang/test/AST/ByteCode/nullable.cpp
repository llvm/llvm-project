// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -verify=ref,both %s


constexpr int dummy = 1;
constexpr const int *null = nullptr;

namespace simple {
  __attribute__((nonnull))
  constexpr int simple1(const int*) {
    return 1;
  }
  static_assert(simple1(&dummy) == 1, "");
  static_assert(simple1(nullptr) == 1, ""); // both-error {{not an integral constant expression}} \
                                            // both-note {{null passed to a callee}}
  static_assert(simple1(null) == 1, ""); // both-error {{not an integral constant expression}} \
                                         // both-note {{null passed to a callee}}

  __attribute__((nonnull)) // both-warning {{applied to function with no pointer arguments}}
  constexpr int simple2(const int &a) {
    return 12;
  }
  static_assert(simple2(1) == 12, "");
}

namespace methods {
  struct S {
    __attribute__((nonnull(2))) // both-warning {{only applies to pointer arguments}}
    __attribute__((nonnull(3)))
    constexpr int foo(int a, const void *p) const {
      return 12;
    }

    __attribute__((nonnull(3)))
    constexpr int foo2(...) const {
      return 12;
    }

    __attribute__((nonnull))
    constexpr int foo3(...) const {
      return 12;
    }
  };

  constexpr S s{};
  static_assert(s.foo(8, &dummy) == 12, "");

  static_assert(s.foo2(nullptr) == 12, "");
  static_assert(s.foo2(1, nullptr) == 12, ""); // both-error {{not an integral constant expression}} \
                                               // both-note {{null passed to a callee}}

  constexpr S *s2 = nullptr;
  static_assert(s2->foo3() == 12, ""); // both-error {{not an integral constant expression}} \
                                       // both-note {{member call on dereferenced null pointer}}
}

namespace fnptrs {
  __attribute__((nonnull))
  constexpr int add(int a, const void *p) {
    return a + 1;
  }
  __attribute__((nonnull(3)))
  constexpr int applyBinOp(int a, int b, int (*op)(int, const void *)) {
    return op(a, nullptr); // both-note {{null passed to a callee}}
  }
  static_assert(applyBinOp(10, 20, add) == 11, ""); // both-error {{not an integral constant expression}} \
                                                    // both-note {{in call to}}

  static_assert(applyBinOp(10, 20, nullptr) == 11, ""); // both-error {{not an integral constant expression}} \
                                                        // both-note {{null passed to a callee}}
}

namespace lambdas {
  auto lstatic = [](const void *P) __attribute__((nonnull)) { return 3; };
  static_assert(lstatic(nullptr) == 3, ""); // both-error {{not an integral constant expression}} \
                                            // both-note {{null passed to a callee}}
}

// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -pedantic %s
// expected-no-diagnostics

namespace PR12866 {
  struct bar {
    union {
      int member;
    };
  };

  void foo( void ) {
    (void)sizeof(bar::member);
  }
}

namespace PR20021 {
class C {
  union {
    static_assert(true, "");
    int i;
  };
};
}

// RUN: %clang_cc1 -verify %s
// expected-no-diagnostics

// PR12223
namespace test1 {
  namespace N {
    extern "C" void f(struct S*);
    void g(S*);
  }
  namespace N {
    void f(struct S *s) {
      g(s);
    }
  }
}

// PR10447
namespace test2 {
  extern "C" {
    void f(struct Bar*) { }
    test2::Bar *ptr;
  }
}

// RUN: %clang_cc1 -fsyntax-only -verify -fcxx-exceptions -Wno-unevaluated-expression -std=c++20 %s

namespace std {
  struct destroying_delete_t {
    explicit destroying_delete_t() = default;
  };

  inline constexpr destroying_delete_t destroying_delete{};
}

struct Explicit {
    ~Explicit() noexcept(false) {}

    void operator delete(Explicit*, std::destroying_delete_t) noexcept {
    }
};

Explicit *qn = nullptr;
// This assertion used to fail, see GH118660
static_assert(noexcept(delete(qn)));

struct ThrowingDestroyingDelete {
    ~ThrowingDestroyingDelete() noexcept(false) {}

    void operator delete(ThrowingDestroyingDelete*, std::destroying_delete_t) noexcept(false) {
    }
};

ThrowingDestroyingDelete *pn = nullptr;
// noexcept should return false here because the destroying delete itself is a
// potentially throwing function.
static_assert(!noexcept(delete(pn)));


struct A {
  virtual ~A(); // implicitly noexcept
};
struct B : A {
  void operator delete(B *p, std::destroying_delete_t) { throw "oh no"; } // expected-warning {{'operator delete' has a non-throwing exception specification but can still throw}} \
                                                                             expected-note {{deallocator has a implicit non-throwing exception specification}}
};
A *p = new B;

// Because the destructor for A is virtual, it is the only thing we consider
// when determining whether the delete expression can throw or not, despite the
// fact that the selected operator delete is a destroying delete. For virtual
// destructors, it's the dynamic type that matters.
static_assert(noexcept(delete p));

// RUN: %clang_cc1 -fsyntax-only -verify -fcxx-exceptions -Wno-unevaluated-expression -std=c++20 %s
// expected-no-diagnostics

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

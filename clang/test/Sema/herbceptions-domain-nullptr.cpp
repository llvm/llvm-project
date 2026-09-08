// RUN: %clang_cc1 -std=c++26 -fherbceptions -fcxx-exceptions -fsyntax-only -verify %s

// std::error_domain<T>::domain() must not return nullptr: the fabricated
// std::error dereferences the domain pointer in ~error(). The trivial
// `return nullptr;` / `return 0;` forms are diagnosed.

namespace std {
struct error_domain_singleton {};
template <typename T> struct error_domain;

enum class bad1 : unsigned { a = 1 };
template <> struct error_domain<bad1> {
  static constexpr error_domain_singleton const *domain() noexcept {
    return nullptr; // expected-error {{std::error_domain<T>::domain() must not return nullptr; the fabricated std::error dereferences the domain pointer}}
  }
  static constexpr __SIZE_TYPE__ code(bad1) noexcept { return 0; }
};

enum class bad2 : unsigned { a = 1 };
template <> struct error_domain<bad2> {
  static constexpr error_domain_singleton const *domain() noexcept {
    return 0; // expected-error {{std::error_domain<T>::domain() must not return nullptr; the fabricated std::error dereferences the domain pointer}}
  }
  static constexpr __SIZE_TYPE__ code(bad2) noexcept { return 0; }
};

namespace {
constinit error_domain_singleton good_domain{};
}

enum class ok : unsigned { a = 1 };
template <> struct error_domain<ok> {
  static constexpr error_domain_singleton const *domain() noexcept {
    return &good_domain;
  }
  static constexpr __SIZE_TYPE__ code(ok) noexcept { return 0; }
};
} // namespace std

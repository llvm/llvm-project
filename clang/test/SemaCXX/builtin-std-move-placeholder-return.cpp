// RUN: %clang_cc1 -std=c++23 -verify %s -DPLACEHOLDER="decltype(auto)"
// RUN: %clang_cc1 -std=c++23 -verify %s -DPLACEHOLDER="auto&&"
// RUN: %clang_cc1 -std=c++23 -verify %s -DPLACEHOLDER="True decltype(auto)"

// expected-no-diagnostics

template<class T>
concept True = true;

namespace std {
  template<class T>
  constexpr PLACEHOLDER move(T&& t) noexcept {
    return static_cast<__remove_reference_t(T)&&>(t);
  }
  constexpr PLACEHOLDER move_if_noexcept(auto& t) noexcept {
    return static_cast<__remove_reference_t(decltype(t))&&>(t);
  }
  template<class T, class U>
  constexpr PLACEHOLDER forward_like(U&& x) noexcept {
    if constexpr (__is_const(__remove_reference_t(T))) {
      using copy_const = const __remove_reference_t(U);
      if constexpr (__is_rvalue_reference(T&&)) {
        using V = __remove_reference_t(copy_const)&&;
        return static_cast<V>(x);
      } else {
        using V = copy_const&;
        return static_cast<V>(x);
      }
    } else {
      using copy_const = __remove_reference_t(U);
      if constexpr (__is_rvalue_reference(T&&)) {
        using V = __remove_reference_t(copy_const)&&;
        return static_cast<V>(x);
      } else {
        using V = copy_const&;
        return static_cast<V>(x);
      }
    }
  }
}

namespace GH101614 {
int i;
static_assert(__is_same(decltype(std::move(i)), int&&));
static_assert(__is_same(decltype(std::move_if_noexcept(i)), int&&));
static_assert(__is_same(decltype(std::forward_like<char>(i)), int&&));
static_assert(__is_same(decltype(std::forward_like<char&>(i)), int&));

constexpr bool is_i(int&& x) { return &x == &i; }
void is_i(int&) = delete;
void is_i(auto&&) = delete;

static_assert(is_i(std::move(i)));
static_assert(is_i(std::move_if_noexcept(i)));
static_assert(is_i(std::forward_like<char>(i)));
static_assert(&std::forward_like<char&>(i) == &i);

// These types are incorrect, but make sure the types as declared are used
static_assert(__is_same(decltype(std::move<int(&)()>), auto(int(&)()) noexcept -> int(&)()));
static_assert(__is_same(decltype(std::move_if_noexcept<int(&)()>), auto(int(&)()) noexcept -> int(&)()));
static_assert(__is_same(decltype(std::forward_like<int, int(&)()>), auto(int(&)()) noexcept -> int(&)()));
} // namespace GH101614

// RUN: %clang_cc1 -verify=expected,both -std=c++26 %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -verify=ref,both      -std=c++26 %s

// both-no-diagnostics

namespace std {
  struct type_info;
  struct destroying_delete_t {
    explicit destroying_delete_t() = default;
  } inline constexpr destroying_delete{};
  struct nothrow_t {
    explicit nothrow_t() = default;
  } inline constexpr nothrow{};
  using size_t = decltype(sizeof(0));
  enum class align_val_t : size_t {};
};

constexpr void *operator new(std::size_t, void *p) { return p; }
namespace std {
  template<typename T> constexpr T *construct_at(T *p) { return new (p) T; }
  template<typename T> constexpr void destroy_at(T *p) { p->~T(); }
}

constexpr bool foo() {
  using T = bool;
  bool b = true;
  b.~T();
  new (&b) bool(false);
  return b;
}
static_assert(!foo());

struct S {};
constexpr bool foo2() {
  S s;
  s.~S();
  new (&s) S{};
  return true;
}
static_assert(foo2());

constexpr void destroy_pointer() {
  using T = int*;
  T p;
  p.~T();
  std::construct_at(&p);
}
static_assert((destroy_pointer(), true));


namespace DestroyArrayElem {
  /// This is proof that std::destroy_at'ing an array element
  /// ends the lifetime of the entire array.
  /// See https://github.com/llvm/llvm-project/issues/147528
  /// Using destroy_at on array elements is currently a no-op due to this.
  constexpr int test() {
    int a[4] = {};

    std::destroy_at(&a[3]);
    int r = a[1];
    std::construct_at(&a[3]);

    return r;
  }
  static_assert(test() == 0);
}

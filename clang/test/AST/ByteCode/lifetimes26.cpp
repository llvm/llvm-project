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
  template<typename T> constexpr T *construct(T *p) { return new (p) T; }
  template<typename T> constexpr void destroy(T *p) { p->~T(); }
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
  std::construct(&p);
}
static_assert((destroy_pointer(), true));


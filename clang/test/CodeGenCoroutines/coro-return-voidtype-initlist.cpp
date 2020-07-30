// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcoroutines-ts -std=c++1z -emit-llvm %s -o - -disable-llvm-passes | FileCheck %s

namespace std {
template <typename a>
struct b { b(int, a); };
template <typename, typename = int>
struct c {};
namespace experimental {
template <typename d>
struct coroutine_traits : d {};
template <typename = void>
struct coroutine_handle;
template <>
struct coroutine_handle<> {};
template <typename>
struct coroutine_handle : coroutine_handle<> {
  static coroutine_handle from_address(void *) noexcept;
};
struct e {
  int await_ready();
  void await_suspend(coroutine_handle<>);
  void await_resume();
};
} // namespace experimental
} // namespace std
template <typename ag>
auto ah(ag) { return ag().ah(0); }
template <typename>
struct f;
struct g {
  struct h {
    int await_ready() noexcept;
    template <typename al>
    void await_suspend(std::experimental::coroutine_handle<al>) noexcept;
    void await_resume() noexcept;
  };
  std::experimental::e initial_suspend();
  h final_suspend() noexcept;
  template <typename ag>
  auto await_transform(ag) { return ah(ag()); }
};
struct j : g {
  f<std::b<std::c<int, int>>> get_return_object();
  void return_value(std::b<std::c<int, int>>);
  void unhandled_exception();
};
struct k {
  k(std::experimental::coroutine_handle<>);
  int await_ready();
};
template <typename am>
struct f {
  using promise_type = j;
  std::experimental::coroutine_handle<> ar;
  struct l : k {
    using at = k;
    l(std::experimental::coroutine_handle<> m) : at(m) {}
    void await_suspend(std::experimental::coroutine_handle<>);
  };
  struct n : l {
    n(std::experimental::coroutine_handle<> m) : l(m) {}
    am await_resume();
  };
  auto ah(int) { return n(ar); }
};
template <typename am, typename av, typename aw>
auto ax(std::c<am, av>, aw) -> f<std::c<int, aw>>;
template <typename>
struct J { static f<std::b<std::c<int, int>>> bo(); };
// CHECK-LABEL: _ZN1JIiE2boEv(
template <typename bc>
f<std::b<std::c<int, int>>> J<bc>::bo() {
  std::c<int> bu;
  int bw(0);
  // CHECK: void @_ZN1j12return_valueESt1bISt1cIiiEE(%struct.j* %__promise)
  co_return{0, co_await ax(bu, bw)};
}
void bh() {
  auto cn = [] { J<int>::bo; };
  cn();
}

// RUN: %check_clang_tidy -std=c++20-or-later %s bugprone-use-after-move %t -- \
// RUN:   -config='{CheckOptions: { \
// RUN:     bugprone-use-after-move.Awaitables: "::std::suspend_always;::CustomAwaitable", \
// RUN:     bugprone-use-after-move.NonlocalAccessors: "::GetNonlocalState" \
// RUN:   }}' -- \
// RUN:   -fno-delayed-template-parsing

namespace std {
template <typename R, typename...>
struct coroutine_traits {
  using promise_type = typename R::promise_type;
};

template <typename Promise = void>
struct coroutine_handle;

template <>
struct coroutine_handle<void> {
  static coroutine_handle from_address(void *addr) noexcept;
  void operator()();
  void *address() const noexcept;
  void resume() const;
  void destroy() const;
  bool done() const;
  coroutine_handle &operator=(decltype(nullptr));
  coroutine_handle(decltype(nullptr));
  coroutine_handle();
  explicit operator bool() const;
};

template <typename Promise>
struct coroutine_handle : coroutine_handle<> {
  using coroutine_handle<>::operator=;
  static coroutine_handle from_address(void *addr) noexcept;
  Promise &promise() const;
  static coroutine_handle from_promise(Promise &promise);
};

struct suspend_always {
  bool await_ready() noexcept;
  void await_suspend(coroutine_handle<>) noexcept;
  void await_resume() noexcept;
};

} // namespace std

struct CustomAwaitable {
  bool await_ready() noexcept;
  void await_suspend(std::coroutine_handle<>) noexcept;
  void await_resume() noexcept;
};

class A {
public:
  A();
  A(const A &);
  A(A &&);

  A &operator=(const A &);
  A &operator=(A &&);

  void foo() const;
  void bar(int i) const;
  int getInt() const;

  operator bool() const;

  int i;
};

template <class Elem, class Final>
class Coroutine final {
 public:
  struct promise_type;
  explicit Coroutine(std::coroutine_handle<promise_type> h);
  ~Coroutine();
  struct promise_type {
    std::suspend_always final_suspend() noexcept;
    Coroutine get_return_object() noexcept;
    std::suspend_always initial_suspend() noexcept;
    void return_void();
    void unhandled_exception();
    std::suspend_always yield_value(const Elem &);
  };
};

struct GlobalState {
  int val() const;
  const int &ref() const;
  const void *ptr() const;
};
const GlobalState &GetNonlocalState();

template <class T>
void use(const T &);

namespace coroutines {

Coroutine<int, void> simpleSuspension() {
  {
    A a;
    a.foo();
    auto &&ctx = GetNonlocalState();
    use(ctx);
    co_yield 0;
    a.foo();
    use(ctx);
    // CHECK-NOTES: [[@LINE-1]]:9: warning: 'ctx' used after a suspension point [bugprone-use-after-move]
    // CHECK-NOTES: [[@LINE-4]]:5: note: suspension occurred here
  }
  {
    A a;
    a.foo();
    auto &ctx = GetNonlocalState().ref();
    use(ctx);
    co_await CustomAwaitable();
    a.foo();
    use(ctx);
    // CHECK-NOTES: [[@LINE-1]]:9: warning: 'ctx' used after a suspension point [bugprone-use-after-move]
    // CHECK-NOTES: [[@LINE-4]]:5: note: suspension occurred here
  }
  {
    A a;
    a.foo();
    auto *ctx = GetNonlocalState().ptr();
    use(ctx);
    co_yield 0;
    a.foo();
    use(ctx);
    // CHECK-NOTES: [[@LINE-1]]:9: warning: 'ctx' used after a suspension point [bugprone-use-after-move]
    // CHECK-NOTES: [[@LINE-4]]:5: note: suspension occurred here
  }
  {
    A a;
    a.foo();
    auto &&ctx = GetNonlocalState().val();
    use(ctx);
    co_yield 0;
    a.foo();
    use(ctx);  // No error
  }
}

} // namespace coroutines

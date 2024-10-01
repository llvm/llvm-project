// RUN: %clang_cc1 -verify -fsyntax-only %s

void func() { // expected-note {{'func' declared here}}
  __builtin_invoke(); // expected-error {{too few arguments to function call, expected at least 1, have 0}}
}

void nfunc() noexcept {}

struct S {};
void argfunc(int, S) {} // expected-note {{'argfunc' declared here}}

struct Callable {
  void operator()() {}

  void func() {}

  int var;
};

void* malloc(decltype(sizeof(int)));

template <class T>
struct pointer_wrapper {
  T* v;

  T& operator*() {
    return *v;
  }
};

namespace std {
  template <class T>
  class reference_wrapper {
    T* ptr;

  public:
    reference_wrapper(T& ref) : ptr(&ref) {}

    T& get() { return *ptr; }
  };

  template <class T>
  reference_wrapper<T> ref(T& v) {
    return reference_wrapper<T>(v);
  }
} // namespace std

struct InvalidSpecialization1 {
  void func() {}

  int var;
};

template <>
class std::reference_wrapper<InvalidSpecialization1> {
public:
  reference_wrapper(InvalidSpecialization1&) {}
};

struct InvalidSpecialization2 {
  void func() {}

  int var;
};

template <>
class std::reference_wrapper<InvalidSpecialization2> {
public:
  reference_wrapper(InvalidSpecialization2&) {}

private:
  InvalidSpecialization2& get(); // expected-note 2 {{declared private here}}
};

void call() {
  __builtin_invoke(func);
  __builtin_invoke(nfunc);
  static_assert(!noexcept(__builtin_invoke(func)));
  static_assert(noexcept(__builtin_invoke(nfunc)));
  __builtin_invoke(func, 1); // expected-error {{too many arguments to function call, expected 0, have 1}}
  __builtin_invoke(argfunc, 1); // expected-error {{too few arguments to function call, expected 2, have 1}}
  __builtin_invoke(Callable{});
  __builtin_invoke(malloc, 0);
  __builtin_invoke(__builtin_malloc, 0); // expected-error {{builtin functions must be directly called}}

  // Member functiom pointer
  __builtin_invoke(&Callable::func); // expected-error {{too few arguments to function call, expected at least 2, have 1}}
  __builtin_invoke(&Callable::func, 1); // expected-error {{indirection requires pointer operand ('int' invalid)}}
  __builtin_invoke(&Callable::func, Callable{});
  __builtin_invoke(&Callable::func, Callable{}, 1); // expected-error {{too many arguments to function call, expected 0, have 1}}

  Callable c;
  __builtin_invoke(&Callable::func, &c);
  __builtin_invoke(&Callable::func, std::ref(c));
  __builtin_invoke(&Callable::func, &c);
  __builtin_invoke(&Callable::func, &c, 2); // expected-error {{too many arguments to function call, expected 0, have 1}}
  __builtin_invoke(&Callable::func, pointer_wrapper<Callable>{&c});
  __builtin_invoke(&Callable::func, pointer_wrapper<Callable>{&c}, 2); // expected-error {{too many arguments to function call, expected 0, have 1}}

  InvalidSpecialization1 is1;
  InvalidSpecialization2 is2;
  __builtin_invoke(&InvalidSpecialization1::func, std::ref(is1)); // expected-error {{no member named 'get' in 'std::reference_wrapper<InvalidSpecialization1>'}}
  __builtin_invoke(&InvalidSpecialization2::func, std::ref(is2)); // expected-error {{'get' is a private member of 'std::reference_wrapper<InvalidSpecialization2>'}}

  // Member data pointer
  __builtin_invoke(&Callable::var); // expected-error {{too few arguments to function call, expected at least 2, have 1}}
  __builtin_invoke(&Callable::var, 1); // expected-error {{indirection requires pointer operand ('int' invalid)}}
  (void)__builtin_invoke(&Callable::var, Callable{});
  __builtin_invoke(&Callable::var, Callable{}, 1); // expected-error {{too many arguments to function call, expected 2, have 3}}

  (void)__builtin_invoke(&Callable::var, &c);
  (void)__builtin_invoke(&Callable::var, std::ref(c));
  (void)__builtin_invoke(&Callable::var, &c);
  __builtin_invoke(&Callable::var, &c, 2); // expected-error {{too many arguments to function call, expected 2, have 3}}
  (void)__builtin_invoke(&Callable::var, pointer_wrapper<Callable>{&c});
  __builtin_invoke(&Callable::var, pointer_wrapper<Callable>{&c}, 2); // expected-error {{too many arguments to function call, expected 2, have 3}}

  __builtin_invoke(&InvalidSpecialization1::var, std::ref(is1)); // expected-error {{no member named 'get' in 'std::reference_wrapper<InvalidSpecialization1>'}}
  (void)__builtin_invoke(&InvalidSpecialization2::var, std::ref(is2)); // expected-error {{'get' is a private member of 'std::reference_wrapper<InvalidSpecialization2>'}}
}

[[nodiscard]] int diagnose_discard();
int no_diagnose_discard();

namespace std {
  template <class... Args>
  auto invoke(Args&&... args) -> decltype(__builtin_invoke(args...));
} // namespace std

void test3() {
  __builtin_invoke(diagnose_discard); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  __builtin_invoke(no_diagnose_discard);
}

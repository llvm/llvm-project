// RUN: %clang_cc1 -verify -fsyntax-only %s -std=c++23

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
    constexpr reference_wrapper(T& ref) : ptr(&ref) {}

    constexpr T& get() { return *ptr; }
  };

  template <class T>
  constexpr reference_wrapper<T> ref(T& v) {
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

struct ExplicitObjectParam {
  void func(this const ExplicitObjectParam& self) {}
};

struct Incomplete; // expected-note 2 {{forward declaration}}
struct Incomplete2;

void incomplete_by_val_test(Incomplete);

void incomplete_test(Incomplete& incomplete) {
  __builtin_invoke((int (Incomplete2::*)){}, incomplete); // expected-error {{incomplete type 'Incomplete' used in type trait expression}} \
                                                             expected-error {{indirection requires pointer operand ('Incomplete' invalid)}}
  __builtin_invoke(incomplete_test, incomplete);
  __builtin_invoke(incomplete_by_val_test, incomplete); // expected-error {{argument type 'Incomplete' is incomplete}}
}

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

  // Variadic function
  void variadic_func(int, ...); // expected-note {{declared here}}

  __builtin_invoke(variadic_func); // expected-error {{too few arguments to function call, expected at least 1, have 0}}
  __builtin_invoke(variadic_func, 1);
  __builtin_invoke(variadic_func, 1, 2, 3);

  // static member function
  struct StaticMember {
    static void func(int);
  };

  __builtin_invoke(StaticMember::func, 1);
  StaticMember sm;
  __builtin_invoke(sm.func, 1);

  // lambda
  __builtin_invoke([] {});
  __builtin_invoke([](int) {}, 1);

  // Member function pointer
  __builtin_invoke(&Callable::func); // expected-error {{too few arguments to function call, expected at least 2, have 1}}
  __builtin_invoke(&Callable::func, 1); // expected-error {{indirection requires pointer operand ('int' invalid)}}
  __builtin_invoke(&Callable::func, Callable{});
  __builtin_invoke(&Callable::func, Callable{}, 1); // expected-error {{too many arguments to function call, expected 0, have 1}}
  __builtin_invoke(&ExplicitObjectParam::func, ExplicitObjectParam{});

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

template <class... Args>
concept invocable = requires(Args... args) { __builtin_invoke(args...); };

static_assert(!invocable<std::reference_wrapper<InvalidSpecialization1>>);
static_assert(!invocable<std::reference_wrapper<InvalidSpecialization2>>);

void test3() {
  __builtin_invoke(diagnose_discard); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  __builtin_invoke(no_diagnose_discard);
}

template <class T>
auto test(T v) {
  return __builtin_invoke(v);
}

auto call2() {
  test(call);
}

template <class ClassT, class FuncT>
void func(ClassT& c, FuncT&& func) {
  __builtin_invoke(func, c, 1, 2, 3); // expected-error {{too many arguments to function call, expected 0, have 3}}
}

struct DependentTest {
  void func(int, int, int);
  void bad_func();
};

void call3() {
  DependentTest d;
  func(d, &DependentTest::func);
  func(d, &DependentTest::bad_func); // expected-note {{requested here}}
}

constexpr int constexpr_func() {
  return 42;
}

struct ConstexprTestStruct {
  int i;
  constexpr int func() {
    return 55;
  }
};

// Make sure that constant evaluation works
static_assert([]() {

  ConstexprTestStruct s;
  if (__builtin_invoke(&ConstexprTestStruct::func, s) != 55) // [func.requires]/p1.1
    return false;
  if (__builtin_invoke(&ConstexprTestStruct::func, std::ref(s)) != 55) // [func.requires]/p1.2
    return false;
  if (__builtin_invoke(&ConstexprTestStruct::func, &s) != 55) // [func.requires]/p1.3
    return false;

  s.i = 22;
  if (__builtin_invoke(&ConstexprTestStruct::i, s) != 22) // [func.requires]/p1.4
    return false;
  if (__builtin_invoke(&ConstexprTestStruct::i, std::ref(s)) != 22) // [func.requires]/p1.5
    return false;
  if (__builtin_invoke(&ConstexprTestStruct::i, &s) != 22) // [func.requires]/p1.6
    return false;

  // [func.requires]/p1.7
  if (__builtin_invoke(constexpr_func) != 42)
    return false;
  if (__builtin_invoke([] { return 34; }) != 34)
    return false;

  return true;
}());

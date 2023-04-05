// RUN: %clang_cc1 -std=c++2b -verify -fsyntax-only %s

template <typename T, typename U>
constexpr bool is_same = false;

template <typename T>
constexpr bool is_same<T, T> = true;

void f() {

  int y;

  static_assert(is_same<const int &,
                        decltype([x = 1] -> decltype((x)) { return x; }())>);

  static_assert(is_same<int &,
                        decltype([x = 1] mutable -> decltype((x)) { return x; }())>);

  static_assert(is_same<const int &,
                        decltype([=] -> decltype((y)) { return y; }())>);

  static_assert(is_same<int &,
                        decltype([=] mutable -> decltype((y)) { return y; }())>);

  static_assert(is_same<const int &,
                        decltype([=] -> decltype((y)) { return y; }())>);

  static_assert(is_same<int &,
                        decltype([=] mutable -> decltype((y)) { return y; }())>);

  auto ref = [&x = y](
                 decltype([&](decltype(x)) { return 0; }) y) {
    return x;
  };
}

void test_noexcept() {

  int y;

  static_assert(noexcept([x = 1] noexcept(is_same<const int &, decltype((x))>) {}()));
  static_assert(noexcept([x = 1] mutable noexcept(is_same<int &, decltype((x))>) {}()));
  static_assert(noexcept([y] noexcept(is_same<const int &, decltype((y))>) {}()));
  static_assert(noexcept([y] mutable noexcept(is_same<int &, decltype((y))>) {}()));
  static_assert(noexcept([=] noexcept(is_same<const int &, decltype((y))>) {}()));
  static_assert(noexcept([=] mutable noexcept(is_same<int &, decltype((y))>) {}()));
  static_assert(noexcept([&] noexcept(is_same<int &, decltype((y))>) {}()));
  static_assert(noexcept([&] mutable noexcept(is_same<int &, decltype((y))>) {}()));
}

template<typename T>
void test_requires() {

  int x;

  [x = 1]() requires is_same<const int &, decltype((x))> {}
  ();
  [x = 1]() mutable requires is_same<int &, decltype((x))> {}
  ();
  [x]() requires is_same<const int &, decltype((x))> {}
  ();
  [x]() mutable requires is_same<int &, decltype((x))> {}
  ();
  [=]() requires is_same<const int &, decltype((x))> {}
  ();
  [=]() mutable requires is_same<int &, decltype((x))> {}
  ();
  [&]() requires is_same<int &, decltype((x))> {}
  ();
  [&]() mutable requires is_same<int &, decltype((x))> {}
  ();
  [&x]() requires is_same<int &, decltype((x))> {}
  ();
  [&x]() mutable requires is_same<int &, decltype((x))> {}
  ();

  [x = 1]() requires is_same<const int &, decltype((x))> {} ();
  [x = 1]() mutable requires is_same<int &, decltype((x))> {} ();
}

void use() {
  test_requires<int>();
}

void err() {
  int y, z;
  (void)[x = 1]<typename T>
  requires(is_same<const int &, decltype((x))>) {};

  (void)[x = 1]<typename T = decltype((x))>{};

  (void)[=]<typename T = decltype((y))>{};

  (void)[z]<typename T = decltype((z))>{};
}

void gnu_attributes() {
  int y;
  (void)[=]() __attribute__((diagnose_if(!is_same<decltype((y)), const int &>, "wrong type", "warning"))){}();
  // expected-warning@-1 {{wrong type}} expected-note@-1{{'diagnose_if' attribute on 'operator()'}}
  (void)[=]() __attribute__((diagnose_if(!is_same<decltype((y)), int &>, "wrong type", "warning"))){}();

  (void)[=]() __attribute__((diagnose_if(!is_same<decltype((y)), int &>, "wrong type", "warning"))) mutable {}();
  (void)[=]() __attribute__((diagnose_if(!is_same<decltype((y)), const int &>, "wrong type", "warning"))) mutable {}();
  // expected-warning@-1 {{wrong type}} expected-note@-1{{'diagnose_if' attribute on 'operator()'}}


  (void)[x=1]() __attribute__((diagnose_if(!is_same<decltype((x)), const int &>, "wrong type", "warning"))){}();
  // expected-warning@-1 {{wrong type}} expected-note@-1{{'diagnose_if' attribute on 'operator()'}}
  (void)[x=1]() __attribute__((diagnose_if(!is_same<decltype((x)), int &>, "wrong type", "warning"))){}();

  (void)[x=1]() __attribute__((diagnose_if(!is_same<decltype((x)), int &>, "wrong type", "warning"))) mutable {}();
  (void)[x=1]() __attribute__((diagnose_if(!is_same<decltype((x)), const int &>, "wrong type", "warning"))) mutable {}();
  // expected-warning@-1 {{wrong type}} expected-note@-1{{'diagnose_if' attribute on 'operator()'}}
}

void nested() {
  int x, y, z;
  (void)[&](
      decltype([&](
                   decltype([=](
                                decltype([&](
                                             decltype([&](decltype(x)) {})) {})) {})) {})){};

  (void)[&](
      decltype([&](
                   decltype([&](
                                decltype([&](
                                             decltype([&](decltype(y)) {})) {})) {})) {})){};

  (void)[=](
      decltype([=](
                   decltype([=](
                                decltype([=](
                                             decltype([&]<decltype(z)> {})) {})) {})) {})){};
}

template <typename T, typename U>
void dependent(U&& u) {
  [&]() requires is_same<decltype(u), T> {}();
}

template <typename T>
void dependent_init_capture(T x = 0) {
  [ y = x + 1, x ]() mutable -> decltype(y + x)
  requires(is_same<decltype((y)), int &>
        && is_same<decltype((x)), int &>) {
    return y;
  }
  ();
  [ y = x + 1, x ]() -> decltype(y + x)
  requires(is_same<decltype((y)), const int &>
        && is_same<decltype((x)), const int &>) {
    return y;
  }
  ();
}

template <typename T, typename...>
struct extract_type {
  using type = T;
};

template <typename... T>
void dependent_variadic_capture(T... x) {
  [... y = x, x... ](auto...) mutable -> typename extract_type<decltype(y)...>::type requires((is_same<decltype((y)), int &> && ...) && (is_same<decltype((x)), int &> && ...)) {
    return 0;
  }
  (x...);
  [... y = x, x... ](auto...) -> typename extract_type<decltype(y)...>::type requires((is_same<decltype((y)), const int &> && ...) && (is_same<decltype((x)), const int &> && ...)) {
    return 0;
  }
  (x...);
}

void test_dependent() {
  int v   = 0;
  int & r = v;
  const int & cr = v;
  dependent<int&>(v);
  dependent<int&>(r);
  dependent<const int&>(cr);
  dependent_init_capture(0);
  dependent_variadic_capture(1, 2, 3, 4);
}

void check_params() {
  int i = 0;
  int &j = i;
  (void)[=](decltype((j)) jp, decltype((i)) ip) {
    static_assert(is_same<const int&, decltype((j))>);
    static_assert(is_same<const int &, decltype((i))>);
    static_assert(is_same<int &, decltype((jp))>);
    static_assert(is_same<int &, decltype((ip))>);
  };

  (void)[=](decltype((j)) jp, decltype((i)) ip) mutable {
    static_assert(is_same<int &, decltype((j))>);
    static_assert(is_same<int &, decltype((i))>);
    static_assert(is_same<int &, decltype((jp))>);
    static_assert(is_same<int &, decltype((ip))>);
    static_assert(is_same<int &, decltype(jp)>);
    static_assert(is_same<int &, decltype(ip)>);
  };

  (void)[a = 0](decltype((a)) ap) mutable {
    static_assert(is_same<int &, decltype((a))>);
    static_assert(is_same<int, decltype(a)>);
    static_assert(is_same<int &, decltype(ap)>);
  };
  (void)[a = 0](decltype((a)) ap) {
    static_assert(is_same<const int &, decltype((a))>);
    static_assert(is_same<int, decltype(a)>);
    static_assert(is_same<int&, decltype((ap))>);
  };
}

template <typename T>
void check_params_tpl() {
  T i = 0;
  T &j = i;
  (void)[=](decltype((j)) jp, decltype((i)) ip) {
    static_assert(is_same<const int&, decltype((j))>);
    static_assert(is_same<const int &, decltype((i))>);
    static_assert(is_same<const int &, decltype((jp))>);
    static_assert(is_same<const int &, decltype((ip))>);
  };

  (void)[=](decltype((j)) jp, decltype((i)) ip) mutable {
    static_assert(is_same<int &, decltype((j))>);
    static_assert(is_same<int &, decltype((i))>);
    static_assert(is_same<int &, decltype((jp))>);
    static_assert(is_same<int &, decltype((ip))>);
    static_assert(is_same<int &, decltype(jp)>);
    static_assert(is_same<int &, decltype(ip)>);
  };

  (void)[a = 0](decltype((a)) ap) mutable {
    static_assert(is_same<int &, decltype((a))>);
    static_assert(is_same<int, decltype(a)>);
    static_assert(is_same<int &, decltype(ap)>);
  };
  (void)[a = 0](decltype((a)) ap) {
    static_assert(is_same<const int &, decltype((a))>);
    static_assert(is_same<int, decltype(a)>);
    static_assert(is_same<int&, decltype((ap))>);
  };
}

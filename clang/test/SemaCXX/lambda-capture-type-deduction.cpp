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

  static_assert(noexcept([&] mutable noexcept(!is_same<int &, decltype((y))>) {}())); // expected-error {{static_assert failed due}}
}

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

  [x = 1]() requires is_same<int &, decltype((x))> {} (); //expected-error {{no matching function for call to object of type}} \
                                                           // expected-note {{candidate function not viable}} \
                                                           // expected-note {{'is_same<int &, decltype((x))>' evaluated to false}}
  [x = 1]() mutable requires is_same<const int &, decltype((x))> {} (); // expected-error {{no matching function for call to object of type}} \
                                                                     // expected-note {{candidate function not viable}} \
                                                                     // expected-note {{'is_same<const int &, decltype((x))>' evaluated to false}}
}

void err() {
  int y, z;                // expected-note 2{{declared here}}
  auto implicit_tpl = [=]( // expected-note {{variable 'y' is captured here}}
                          decltype(
                              [&]<decltype((y))> { return 0; }) y) { // expected-error{{captured variable 'y' cannot appear here}}
    return y;
  };

  auto init_tpl = [x = 1](                                            // expected-note{{explicitly captured here}}
                      decltype([&]<decltype((x))> { return 0; }) y) { // expected-error {{captured variable 'x' cannot appear here}}
    return x;
  };

  auto implicit = [=]( // expected-note {{variable 'z' is captured here}}
                      decltype(
                          [&](decltype((z))) { return 0; }) z) { // expected-error{{captured variable 'z' cannot appear here}}
    return z;
  };

  auto init = [x = 1](                                            // expected-note{{explicitly captured here}}
                  decltype([&](decltype((x))) { return 0; }) y) { // expected-error {{captured variable 'x' cannot appear here}}
    return x;
  };

  auto use_before_params = [x = 1]<typename T>  // expected-note {{variable 'x' is explicitly captured here}}
  requires(is_same<const int &, decltype((x))>) // expected-error {{captured variable 'x' cannot appear here}}
  {};

  auto use_before_params2 = [x = 1]<typename T =  decltype((x))>  // expected-note {{variable 'x' is explicitly captured here}} \
                                                                 // expected-error {{captured variable 'x' cannot appear here}}
  {};
}

void gnu_attributes() {
  int y;
  (void)[=]() __attribute__((diagnose_if(!is_same<decltype((y)), const int &>, "wrong type", "warning"))){}();
  (void)[=]() __attribute__((diagnose_if(!is_same<decltype((y)), int &>, "wrong type", "warning"))){}();
  // expected-warning@-1 {{wrong type}} expected-note@-1{{'diagnose_if' attribute on 'operator()'}}

  (void)[=]() __attribute__((diagnose_if(!is_same<decltype((y)), int &>, "wrong type", "warning"))) mutable {}();
  (void)[=]() __attribute__((diagnose_if(!is_same<decltype((y)), const int &>, "wrong type", "warning"))) mutable {}();
  // expected-warning@-1 {{wrong type}} expected-note@-1{{'diagnose_if' attribute on 'operator()'}}

  (void)[x=1]() __attribute__((diagnose_if(!is_same<decltype((x)), const int &>, "wrong type", "warning"))){}();
  (void)[x=1]() __attribute__((diagnose_if(!is_same<decltype((x)), int &>, "wrong type", "warning"))){}();
  // expected-warning@-1 {{wrong type}} expected-note@-1{{'diagnose_if' attribute on 'operator()'}}

  (void)[x=1]() __attribute__((diagnose_if(!is_same<decltype((x)), int &>, "wrong type", "warning"))) mutable {}();
  (void)[x=1]() __attribute__((diagnose_if(!is_same<decltype((x)), const int &>, "wrong type", "warning"))) mutable {}();
  // expected-warning@-1 {{wrong type}} expected-note@-1{{'diagnose_if' attribute on 'operator()'}}
}

void nested() {
  int x, y, z; // expected-note {{'x' declared here}} expected-note {{'z' declared here}}
  (void)[&](
      decltype([&](
                   decltype([=]( // expected-note {{variable 'x' is captured here}}
                                decltype([&](
                                             decltype([&](decltype((x))) {}) // expected-error{{captured variable 'x' cannot appear here}}
                                         ) {})) {})) {})){};

  (void)[&](
      decltype([&](
                   decltype([&](
                                decltype([&](
                                             decltype([&](decltype((y))) {})) {})) {})) {})){};

  (void)[=](
      decltype([=](
                   decltype([=](
                                decltype([=](                                // expected-note {{variable 'z' is captured here}}
                                             decltype([&]<decltype((z))> {}) // expected-error{{captured variable 'z' cannot appear here}}
                                         ) {})) {})) {})){};
}

template <typename T, typename U>
void dependent(U&& u) {
  [&]() requires is_same<decltype(u), T> {}();
}

void test_dependent() {
  int v   = 0;
  int & r = v;
  const int & cr = v;
  dependent<int&>(v);
  dependent<int&>(r);
  dependent<const int&>(cr);
}

void test_CWG2569_tpl(auto a) {
  (void)[=]<typename T = decltype(a)>(decltype(a) b = decltype(a)()){};
}

void test_CWG2569() {
  int a = 0;
  (void)[=]<typename T = decltype(a)>(decltype(a) b = decltype(a)()){};
  test_CWG2569_tpl(0);

  (void)[=]<typename T = decltype(not_a_thing)>(decltype(not_a_thing)){}; // expected-error 2{{use of undeclared identifier 'not_a_thing'}}
}

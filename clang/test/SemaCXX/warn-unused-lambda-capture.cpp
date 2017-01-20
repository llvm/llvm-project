// RUN: %clang_cc1 -fsyntax-only -Wunused-lambda-capture -Wused-but-marked-unused -Wno-uninitialized -verify -std=c++14 %s

class NonTrivialConstructor {
public:
  NonTrivialConstructor() {}
};

class NonTrivialDestructor {
public:
  ~NonTrivialDestructor() {}
};

class Trivial {
public:
  Trivial() = default;
  Trivial(int a) {}
};

int side_effect() {
  return 42;
}

void test() {
  int i = 0;
  const int k = 0;

  auto captures_nothing = [] {};

  auto captures_nothing_by_value = [=] {};
  auto captures_nothing_by_reference = [&] {};

  auto implicit_by_value = [=]() mutable { i++; };
  auto implicit_by_reference = [&] { i++; };

  auto explicit_by_value_used = [i] { return i + 1; };
  auto explicit_by_value_used_void = [i] { (void)i; };
  auto explicit_by_value_unused = [i] {}; // expected-warning{{lambda capture 'i' is not used}}
  auto explicit_by_value_unused_sizeof = [i] { return sizeof(i); }; // expected-warning{{lambda capture 'i' is not required to be captured for this use}}
  auto explicit_by_value_unused_decltype = [i] { decltype(i) j = 0; }; // expected-warning{{lambda capture 'i' is not required to be captured for this use}}
  auto explicit_by_value_unused_const = [k] { return k + 1; };         // expected-warning{{lambda capture 'k' is not required to be captured for this use}}

  auto explicit_by_reference_used = [&i] { i++; };
  auto explicit_by_reference_unused = [&i] {}; // expected-warning{{lambda capture 'i' is not used}}

  auto explicit_initialized_reference_used = [&j = i] { return j + 1; };
  auto explicit_initialized_reference_unused = [&j = i]{}; // expected-warning{{lambda capture 'j' is not used}}

  auto explicit_initialized_value_used = [j = 1] { return j + 1; };
  auto explicit_initialized_value_unused = [j = 1] {}; // expected-warning{{lambda capture 'j' is not used}}
  auto explicit_initialized_value_non_trivial_constructor = [j = NonTrivialConstructor()]{};
  auto explicit_initialized_value_non_trivial_destructor = [j = NonTrivialDestructor()]{};
  auto explicit_initialized_value_trivial_init = [j = Trivial()]{}; // expected-warning{{lambda capture 'j' is not used}}
  auto explicit_initialized_value_non_trivial_init = [j = Trivial(42)]{};
  auto explicit_initialized_value_with_side_effect = [j = side_effect()]{};

  auto nested = [&i] {
    auto explicit_by_value_used = [i] { return i + 1; };
    auto explicit_by_value_unused = [i] {}; // expected-warning{{lambda capture 'i' is not used}}
  };
}

class Foo
{
  void test() {
    auto explicit_this_used = [this] { return i; };
    auto explicit_this_used_void = [this] { (void)this; };
    auto explicit_this_unused = [this] {}; // expected-warning{{lambda capture 'this' is not used}}
  }
  int i;
};

template <typename T>
void test_templated() {
  int i = 0;
  const int k = 0;

  auto captures_nothing = [] {};

  auto captures_nothing_by_value = [=] {};
  auto captures_nothing_by_reference = [&] {};

  auto implicit_by_value = [=]() mutable { i++; };
  auto implicit_by_reference = [&] { i++; };

  auto explicit_by_value_used = [i] { return i + 1; };
  auto explicit_by_value_used_void = [i] { (void)i; };
  auto explicit_by_value_unused = [i] {}; // expected-warning{{lambda capture 'i' is not used}}
  auto explicit_by_value_unused_sizeof = [i] { return sizeof(i); }; // expected-warning{{lambda capture 'i' is not required to be captured for this use}}
  auto explicit_by_value_unused_decltype = [i] { decltype(i) j = 0; }; // expected-warning{{lambda capture 'i' is not used}}
  auto explicit_by_value_unused_const = [k] { return k + 1; };         // expected-warning{{lambda capture 'k' is not required to be captured for this use}}

  auto explicit_by_reference_used = [&i] { i++; };
  auto explicit_by_reference_unused = [&i] {}; // expected-warning{{lambda capture 'i' is not used}}

  auto explicit_initialized_reference_used = [&j = i] { return j + 1; };
  auto explicit_initialized_reference_unused = [&j = i]{}; // expected-warning{{lambda capture 'j' is not used}}

  auto explicit_initialized_value_used = [j = 1] { return j + 1; };
  auto explicit_initialized_value_unused = [j = 1] {}; // expected-warning{{lambda capture 'j' is not used}}
  auto explicit_initialized_value_non_trivial_constructor = [j = NonTrivialConstructor()]{};
  auto explicit_initialized_value_non_trivial_destructor = [j = NonTrivialDestructor()]{};
  auto explicit_initialized_value_trivial_init = [j = Trivial()]{}; // expected-warning{{lambda capture 'j' is not used}}
  auto explicit_initialized_value_non_trivial_init = [j = Trivial(42)]{};
  auto explicit_initialized_value_with_side_effect = [j = side_effect()]{};

  auto nested = [&i] {
    auto explicit_by_value_used = [i] { return i + 1; };
    auto explicit_by_value_unused = [i] {}; // expected-warning{{lambda capture 'i' is not used}}
  };
}

void test_use_template() {
  test_templated<int>(); // expected-note{{in instantiation of function template specialization 'test_templated<int>' requested here}}
}

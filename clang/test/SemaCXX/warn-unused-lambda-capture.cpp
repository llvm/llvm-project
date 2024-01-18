// RUN: %clang_cc1 -fsyntax-only -Wunused-lambda-capture -Wused-but-marked-unused -Wno-uninitialized -verify -std=c++1z %s

class NonTrivialConstructor {
public:
  NonTrivialConstructor() {}
};

class NonTrivialCopyConstructor {
public:
  NonTrivialCopyConstructor() = default;
  NonTrivialCopyConstructor(const NonTrivialCopyConstructor &) {}
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

  Trivial trivial;
  auto explicit_by_value_trivial = [trivial] {}; // expected-warning{{lambda capture 'trivial' is not used}}

  NonTrivialConstructor cons;
  auto explicit_by_value_non_trivial_constructor = [cons] {}; // expected-warning{{lambda capture 'cons' is not used}}

  NonTrivialCopyConstructor copy_cons;
  auto explicit_by_value_non_trivial_copy_constructor = [copy_cons] {};

  NonTrivialDestructor dest;
  auto explicit_by_value_non_trivial_destructor = [dest] {};

  volatile int v;
  auto explicit_by_value_volatile = [v] {};
}

class TrivialThis : Trivial {
  void test() {
    auto explicit_this_used = [this] { return i; };
    auto explicit_this_used_void = [this] { (void)this; };
    auto explicit_this_unused = [this] {}; // expected-warning{{lambda capture 'this' is not used}}
    auto explicit_star_this_used = [*this] { return i; };
    auto explicit_star_this_used_void = [*this] { (void)this; };
    auto explicit_star_this_unused = [*this] {}; // expected-warning{{lambda capture 'this' is not used}}
  }
  int i;
};

class NonTrivialConstructorThis : NonTrivialConstructor {
  void test() {
    auto explicit_this_used = [this] { return i; };
    auto explicit_this_used_void = [this] { (void)this; };
    auto explicit_this_unused = [this] {}; // expected-warning{{lambda capture 'this' is not used}}
    auto explicit_star_this_used = [*this] { return i; };
    auto explicit_star_this_used_void = [*this] { (void)this; };
    auto explicit_star_this_unused = [*this] {}; // expected-warning{{lambda capture 'this' is not used}}
  }
  int i;
};

class NonTrivialCopyConstructorThis : NonTrivialCopyConstructor {
  void test() {
    auto explicit_this_used = [this] { return i; };
    auto explicit_this_used_void = [this] { (void)this; };
    auto explicit_this_unused = [this] {}; // expected-warning{{lambda capture 'this' is not used}}
    auto explicit_star_this_used = [*this] { return i; };
    auto explicit_star_this_used_void = [*this] { (void)this; };
    auto explicit_star_this_unused = [*this] {};
  }
  int i;
};

class NonTrivialDestructorThis : NonTrivialDestructor {
  void test() {
    auto explicit_this_used = [this] { return i; };
    auto explicit_this_used_void = [this] { (void)this; };
    auto explicit_this_unused = [this] {}; // expected-warning{{lambda capture 'this' is not used}}
    auto explicit_star_this_used = [*this] { return i; };
    auto explicit_star_this_used_void = [*this] { (void)this; };
    auto explicit_star_this_unused = [*this] {};
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
  auto explicit_by_value_used_generic = [i](auto c) { return i + 1; };
  auto explicit_by_value_used_void = [i] { (void)i; };

  auto explicit_by_value_unused = [i] {}; // expected-warning{{lambda capture 'i' is not used}} expected-note {{substituting into a lambda}}
  auto explicit_by_value_unused_sizeof = [i] { return sizeof(i); }; // expected-warning{{lambda capture 'i' is not required to be captured for this use}} expected-note {{substituting into a lambda}}
  auto explicit_by_value_unused_decltype = [i] { decltype(i) j = 0; }; // expected-warning{{lambda capture 'i' is not used}} expected-note {{substituting into a lambda}}
  auto explicit_by_value_unused_const = [k] { return k + 1; };         // expected-warning{{lambda capture 'k' is not required to be captured for this use}} expected-note {{substituting into a lambda}}
  auto explicit_by_value_unused_const_generic = [k](auto c) { return k + 1; }; // expected-warning{{lambda capture 'k' is not required to be captured for this use}} expected-note {{substituting into a lambda}}

  auto explicit_by_reference_used = [&i] { i++; };
  auto explicit_by_reference_unused = [&i] {}; // expected-warning{{lambda capture 'i' is not used}} expected-note {{substituting into a lambda}}

  auto explicit_initialized_reference_used = [&j = i] { return j + 1; };
  auto explicit_initialized_reference_unused = [&j = i]{}; // expected-warning{{lambda capture 'j' is not used}} expected-note {{substituting into a lambda}}

  auto explicit_initialized_value_used = [j = 1] { return j + 1; };
  auto explicit_initialized_value_unused = [j = 1] {}; // expected-warning{{lambda capture 'j' is not used}} expected-note {{substituting into a lambda}}
  auto explicit_initialized_value_non_trivial_constructor = [j = NonTrivialConstructor()]{};
  auto explicit_initialized_value_non_trivial_destructor = [j = NonTrivialDestructor()]{};
  auto explicit_initialized_value_trivial_init = [j = Trivial()]{}; // expected-warning{{lambda capture 'j' is not used}} expected-note {{substituting into a lambda}}
  auto explicit_initialized_value_non_trivial_init = [j = Trivial(42)]{};
  auto explicit_initialized_value_with_side_effect = [j = side_effect()]{};
  auto explicit_initialized_value_generic_used = [i = 1](auto c) mutable { i++; };
  auto explicit_initialized_value_generic_unused = [i = 1](auto c) mutable {}; // expected-warning{{lambda capture 'i' is not used}} expected-note {{substituting into a lambda}}

  auto nested = [&i] { // expected-note {{substituting into a lambda}}
    auto explicit_by_value_used = [i] { return i + 1; };
    auto explicit_by_value_unused = [i] {}; // expected-warning{{lambda capture 'i' is not used}} expected-note {{substituting into a lambda}}
  };

  Trivial trivial;
  auto explicit_by_value_trivial = [trivial] {}; // expected-warning{{lambda capture 'trivial' is not used}} expected-note {{substituting into a lambda}}

  NonTrivialConstructor cons;
  auto explicit_by_value_non_trivial_constructor = [cons] {}; // expected-warning{{lambda capture 'cons' is not used}} expected-note {{substituting into a lambda}}

  NonTrivialCopyConstructor copy_cons;
  auto explicit_by_value_non_trivial_copy_constructor = [copy_cons] {};

  NonTrivialDestructor dest;
  auto explicit_by_value_non_trivial_destructor = [dest] {};

  volatile int v;
  auto explicit_by_value_volatile = [v] {};
}

void test_use_template() {
  test_templated<int>(); // expected-note 13{{in instantiation of function template specialization 'test_templated<int>' requested here}}
}

namespace pr35555 {
int a; // expected-note {{declared here}}
void b() {
  int c[a]; // expected-warning {{variable length arrays in C++ are a Clang extension}} \
               expected-note {{read of non-const variable 'a' is not allowed in a constant expression}}
  auto vla_used = [&c] { return c[0]; };
  auto vla_unused = [&c] {}; // expected-warning{{lambda capture 'c' is not used}}
}
} // namespace pr35555

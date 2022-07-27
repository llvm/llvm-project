// RUN: %check_clang_tidy %s bugprone-suspicious-memory-comparison %t \
// RUN: -- -- -target x86_64-unknown-unknown

namespace std {
typedef __SIZE_TYPE__ size_t;
int memcmp(const void *lhs, const void *rhs, size_t count);
} // namespace std

namespace sei_cert_example_oop57_cpp {
class C {
  int i;

public:
  virtual void f();
};

void f(C &c1, C &c2) {
  if (!std::memcmp(&c1, &c2, sizeof(C))) {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: comparing object representation of non-standard-layout type 'C'; consider using a comparison operator instead
  }
}
} // namespace sei_cert_example_oop57_cpp

namespace inner_padding_64bit_only {
struct S {
  int x;
  int *y;
};

void test() {
  S a, b;
  std::memcmp(&a, &b, sizeof(S));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: comparing object representation of type 'S' which does not have a unique object representation; consider comparing the members of the object manually
}
} // namespace inner_padding_64bit_only

namespace padding_in_base {
class Base {
  char c;
  int i;
};

class Derived : public Base {};

class Derived2 : public Derived {};

void testDerived() {
  Derived a, b;
  std::memcmp(&a, &b, sizeof(Base));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: comparing object representation of type 'Derived' which does not have a unique object representation; consider comparing the members of the object manually
  std::memcmp(&a, &b, sizeof(Derived));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: comparing object representation of type 'Derived' which does not have a unique object representation; consider comparing the members of the object manually
}

void testDerived2() {
  Derived2 a, b;
  std::memcmp(&a, &b, sizeof(Base));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: comparing object representation of type 'Derived2' which does not have a unique object representation; consider comparing the members of the object manually
  std::memcmp(&a, &b, sizeof(Derived2));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: comparing object representation of type 'Derived2' which does not have a unique object representation; consider comparing the members of the object manually
}

} // namespace padding_in_base

namespace no_padding_in_base {
class Base {
  int a, b;
};

class Derived : public Base {};

class Derived2 : public Derived {};

void testDerived() {
  Derived a, b;
  std::memcmp(&a, &b, sizeof(Base));
  std::memcmp(&a, &b, sizeof(Derived));
}

void testDerived2() {
  Derived2 a, b;
  std::memcmp(&a, &b, sizeof(char));
  std::memcmp(&a, &b, sizeof(Base));
  std::memcmp(&a, &b, sizeof(Derived2));
}
} // namespace no_padding_in_base

namespace non_standard_layout {
class C {
private:
  int x;

public:
  int y;
};

void test() {
  C a, b;
  std::memcmp(&a, &b, sizeof(C));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: comparing object representation of non-standard-layout type 'C'; consider using a comparison operator instead
}

} // namespace non_standard_layout

namespace static_ignored {
struct S {
  static char c;
  int i;
};

void test() {
  S a, b;
  std::memcmp(&a, &b, sizeof(S));
}
} // namespace static_ignored

namespace operator_void_ptr {
struct S {
  operator void *() const;
};

void test() {
  S s;
  std::memcmp(s, s, sizeof(s));
}
} // namespace operator_void_ptr

namespace empty_struct {
struct S {};

void test() {
  S a, b;
  std::memcmp(&a, &b, sizeof(S));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: comparing object representation of type 'S' which does not have a unique object representation; consider comparing the members of the object manually
}
} // namespace empty_struct

namespace empty_field {
struct Empty {};
struct S {
  Empty e;
};

void test() {
  S a, b;
  std::memcmp(&a, &b, sizeof(S));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: comparing object representation of type 'S' which does not have a unique object representation; consider comparing the members of the object manually
}
} // namespace empty_field

namespace no_unique_address_attribute {
struct Empty {};

namespace no_padding {
struct S {
  char c;
  [[no_unique_address]] Empty e;
};

void test() {
  S a, b;
  std::memcmp(&a, &b, sizeof(S));
}

} // namespace no_padding

namespace multiple_empties_same_type {
struct S {
  char c;
  [[no_unique_address]] Empty e1, e2;
};

void test() {
  S a, b;
  std::memcmp(&a, &b, sizeof(S));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: comparing object representation of type 'S' which does not have a unique object representation; consider comparing the members of the object manually
}

} // namespace multiple_empties_same_type

namespace multiple_empties_different_types {
struct Empty2 {};

struct S {
  char c;
  [[no_unique_address]] Empty e1;
  [[no_unique_address]] Empty2 e2;
};

void test() {
  S a, b;
  std::memcmp(&a, &b, sizeof(S));
}
} // namespace multiple_empties_different_types
} // namespace no_unique_address_attribute

namespace alignment {
struct S {
  char x;
  alignas(sizeof(int)) char y[sizeof(int)];
};

void test() {
  S a, b;
  std::memcmp(&a, &b, sizeof(S));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: comparing object representation of type 'S' which does not have a unique object representation; consider comparing the members of the object manually
}
} // namespace alignment

namespace no_warning_in_template {
template <typename T>
int compare(const T *l, const T *r) {
  return std::memcmp(l, r, sizeof(T));
}

void test() {
  int a, b;
  compare(&a, &b);
}
} // namespace no_warning_in_template

namespace warning_in_template {
template <typename T>
int compare(const T *l, const T *r) {
  return std::memcmp(l, r, sizeof(T));
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: comparing object representation of type 'float' which does not have a unique object representation; consider comparing the values manually
}

void test() {
  float a, b;
  compare(&a, &b);
}
} // namespace warning_in_template

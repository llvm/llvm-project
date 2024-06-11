// RUN: %clang_cc1 -fsyntax-only -std=c++20 -verify=expected -Wc++20-extensions %s
// RUN: %clang_cc1 -fsyntax-only -std=c++17 -verify=expected,cxx11-17 -Wc++17-extensions %s
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify=expected,cxx11-17,cxx11 -Wc++17-extensions -Wc++20-extensions %s

struct [[nodiscard]] S {};
S get_s();
S& get_s_ref();

enum [[nodiscard]] E {};
E get_e();

[[nodiscard]] int get_i();
[[nodiscard]] volatile int &get_vi();

void f() {
  get_s(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  get_i(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  get_vi(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  get_e(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // Okay, warnings are not encouraged
  get_s_ref();
  (void)get_s();
  (void)get_i();
  (void)get_vi();
  (void)get_e();
}

[[nodiscard]] volatile char &(*fp)(); // expected-warning {{'nodiscard' attribute only applies to functions, classes, or enumerations}}
void g() {
  fp(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // OK, warning suppressed.
  (void)fp();
}

namespace PR31526 {
typedef E (*fp1)();
typedef S (*fp2)();

typedef S S_alias;
typedef S_alias (*fp3)();

typedef fp2 fp2_alias;

void f() {
  fp1 one;
  fp2 two;
  fp3 three;
  fp2_alias four;

  one(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  two(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  three(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  four(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // These are all okay because of the explicit cast to void.
  (void)one();
  (void)two();
  (void)three();
  (void)four();
}
} // namespace PR31526

struct [[nodiscard("reason")]] ReasonStruct {};
struct LaterReason;
struct [[nodiscard("later reason")]] LaterReason {};

ReasonStruct get_reason();
LaterReason get_later_reason();
[[nodiscard("another reason")]] int another_reason();

[[nodiscard("conflicting reason")]] int conflicting_reason();
[[nodiscard("special reason")]] int conflicting_reason();

void cxx20_use() {
  get_reason(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute: reason}}
  get_later_reason(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute: later reason}}
  another_reason(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute: another reason}}
  conflicting_reason(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute: special reason}}
}

namespace p1771 {
struct[[nodiscard("Don't throw me away!")]] ConvertTo{};
struct S {
  [[nodiscard]] S();
  [[nodiscard("Don't let that S-Char go!")]] S(char);
  S(int);
  [[gnu::warn_unused_result]] S(double);
  operator ConvertTo();
  [[nodiscard]] operator int();
  [[nodiscard("Don't throw away as a double")]] operator double();
};

struct[[nodiscard("Don't throw me away either!")]] Y{};

void usage() {
  S();    // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
  S('A'); // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute: Don't let that S-Char go!}}
  S(1);
  S(2.2);
  Y(); // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute: Don't throw me away either!}}
  S s;
  ConvertTo{}; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute: Don't throw me away!}}

// AST is different in C++20 mode, pre-2017 a move ctor for ConvertTo is there
// as well, hense the constructor warning.
#if __cplusplus >= 201703L
// expected-warning@+4 {{ignoring return value of function declared with 'nodiscard' attribute: Don't throw me away!}}
#else
// expected-warning@+2 {{ignoring temporary created by a constructor declared with 'nodiscard' attribute: Don't throw me away!}}
#endif
  (ConvertTo) s;
  (int)s; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  (S)'c'; // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute: Don't let that S-Char go!}}
#if __cplusplus >= 201703L
// expected-warning@+4 {{ignoring return value of function declared with 'nodiscard' attribute: Don't throw me away!}}
#else
// expected-warning@+2 {{ignoring temporary created by a constructor declared with 'nodiscard' attribute: Don't throw me away!}}
#endif
  static_cast<ConvertTo>(s);
  static_cast<int>(s); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  static_cast<double>(s); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute: Don't throw away as a double}}
}
}; // namespace p1771

namespace discarded_member_access {
struct X {
  union {
    int variant_member;
  };
  struct {
    int anonymous_struct_member;
  };
  int data_member;
  static int static_data_member;
  enum {
    unscoped_enum
  };
  enum class scoped_enum_t {
    scoped_enum
  };
  using enum scoped_enum_t;
  // cxx11-17-warning@-1 {{using enum declaration is a C++20 extension}}

  void implicit_object_member_function();
  static void static_member_function();
#if __cplusplus >= 202302L
  void explicit_object_member_function(this X self);
#endif
};

[[nodiscard]] X get_X();
// cxx11-warning@-1 {{use of the 'nodiscard' attribute is a C++17 extension}}
void f() {
  (void) get_X().variant_member;
  (void) get_X().anonymous_struct_member;
  (void) get_X().data_member;
  (void) get_X().static_data_member;
  // expected-warning@-1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  (void) get_X().unscoped_enum;
  // expected-warning@-1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  (void) get_X().scoped_enum;
  // expected-warning@-1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  (void) get_X().implicit_object_member_function();
  (void) get_X().static_member_function();
  // expected-warning@-1 {{ignoring return value of function declared with 'nodiscard' attribute}}
#if __cplusplus >= 202302L
  (void) get_X().explicit_object_member_function();
#endif
}
} // namespace discarded_member_access


// cxx11-warning@5 {{use of the 'nodiscard' attribute is a C++17 extension}}
// cxx11-warning@9 {{use of the 'nodiscard' attribute is a C++17 extension}}
// cxx11-warning@12 {{use of the 'nodiscard' attribute is a C++17 extension}}
// cxx11-warning@13 {{use of the 'nodiscard' attribute is a C++17 extension}}
// cxx11-warning@29 {{use of the 'nodiscard' attribute is a C++17 extension}}
// cxx11-warning@65 {{use of the 'nodiscard' attribute is a C++20 extension}}
// cxx11-warning@67 {{use of the 'nodiscard' attribute is a C++20 extension}}
// cxx11-warning@71 {{use of the 'nodiscard' attribute is a C++20 extension}}
// cxx11-warning@73 {{use of the 'nodiscard' attribute is a C++20 extension}}
// cxx11-warning@74 {{use of the 'nodiscard' attribute is a C++20 extension}}
// cxx11-warning@84 {{use of the 'nodiscard' attribute is a C++20 extension}}
// cxx11-warning@86 {{use of the 'nodiscard' attribute is a C++17 extension}}
// cxx11-warning@87 {{use of the 'nodiscard' attribute is a C++20 extension}}
// cxx11-warning@91 {{use of the 'nodiscard' attribute is a C++17 extension}}
// cxx11-warning@92 {{use of the 'nodiscard' attribute is a C++20 extension}}
// cxx11-warning@95 {{use of the 'nodiscard' attribute is a C++20 extension}}

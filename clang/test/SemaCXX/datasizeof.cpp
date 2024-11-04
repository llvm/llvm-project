// RUN: %clang_cc1 -fsyntax-only -triple x86_64-linux-gnu -verify %s

#if !__has_extension(datasizeof)
#  error "Expected datasizeof extension"
#endif

struct HasPadding {
  int i;
  char c;
};

struct HasUsablePadding {
  int i;
  char c;

  HasUsablePadding() {}
};

struct Incomplete; // expected-note {{forward declaration of 'Incomplete'}}

static_assert(__datasizeof(int) == 4);
static_assert(__datasizeof(HasPadding) == 8);
static_assert(__datasizeof(HasUsablePadding) == 5);
static_assert(__datasizeof(void)); // expected-error {{invalid application of '__datasizeof' to an incomplete type 'void'}}
static_assert(__datasizeof(Incomplete)); // expected-error {{invalid application of '__datasizeof' to an incomplete type 'Incomplete'}}

static_assert([] {
  int* p = nullptr;
  HasPadding* p2 = nullptr;
  HasUsablePadding* p3 = nullptr;
  static_assert(__datasizeof(*p) == 4);
  static_assert(__datasizeof *p == 4);
  static_assert(__datasizeof(*p2) == 8);
  static_assert(__datasizeof(*p3) == 5);

  return true;
}());

template <typename Ty>
constexpr int data_size_of() {
  return __datasizeof(Ty);
}
static_assert(data_size_of<int>() == __datasizeof(int));
static_assert(data_size_of<HasPadding>() == __datasizeof(HasPadding));
static_assert(data_size_of<HasUsablePadding>() == __datasizeof(HasUsablePadding));

struct S {
  int i = __datasizeof(S);
  float f;
  char c;
};

static_assert(S{}.i == 9);

namespace GH80284 {
struct Bar; // expected-note{{forward declaration}}
struct Foo {
  Bar x; // expected-error{{field has incomplete type}}
};
constexpr int a = __datasizeof(Foo);
}

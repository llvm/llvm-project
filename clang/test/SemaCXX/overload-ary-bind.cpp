// RUN: %clang_cc1 -std=c++20 -verify %s
// RUN: %clang_cc1 -std=c++17 -verify %s

namespace One {
char (&b(int(&&)[1]))[1]; // #1 expected-note{{too many initializers}}
char (&b(int(&&)[2]))[2]; // #2 expected-note{{too many initializers}}

void f() {
  static_assert(sizeof(b({1})) == 1);    // #1
  static_assert(sizeof(b({1, 2})) == 2); // #2

  b({1, 2, 3}); // expected-error{{no matching function}}
}
} // namespace One

namespace Two {
struct Bob {
  Bob(int = 1);
};

char (&b(Bob(&&)[1]))[1]; // #1
char (&b(Bob(&&)[2]))[2]; // #2

void f() {
  static_assert(sizeof(b({})) == 1);         // #1
  static_assert(sizeof(b({Bob()})) == 1);    // #1
  static_assert(sizeof(b({2, Bob()})) == 2); // #2
}
} // namespace Two

namespace Three {
struct Kevin {
  Kevin(int);
};

char (&b(Kevin(&&)[2]))[2]; // #2 expected-note{{too few initializers}}

void f() {
  b({2}); // #1 expected-error{{no matching function}}
}
} // namespace Three

namespace Four {
char (&b(int(&&)[1], float))[1];  // #1 expected-note{{candidate}}
char (&b(int(&&)[1], double))[2]; // #2 expected-note{{candidate}}

char (&c(float, int(&&)[1]))[1];  // #1 expected-note{{candidate}}
char (&c(double, int(&&)[1]))[2]; // #2 expected-note{{candidate}}

void f() {
  b({1}, 0); // expected-error{{is ambiguous}}
  c(0, {1}); // expected-error{{is ambiguous}}
}
} // namespace Four

typedef decltype(sizeof(char)) size_t;
namespace std {
// sufficient initializer list
template <class _E>
class initializer_list {
  const _E *__begin_;
  size_t __size_;

  constexpr initializer_list(const _E *__b, size_t __s)
      : __begin_(__b),
        __size_(__s) {}

public:
  typedef _E value_type;
  typedef const _E &reference;
  typedef const _E &const_reference;
  typedef size_t size_type;

  typedef const _E *iterator;
  typedef const _E *const_iterator;

  constexpr initializer_list() : __begin_(nullptr), __size_(0) {}

  constexpr size_t size() const { return __size_; }
  constexpr const _E *begin() const { return __begin_; }
  constexpr const _E *end() const { return __begin_ + __size_; }
};
} // namespace std

namespace Five {
struct ugly {
  ugly(char *);
  ugly(int);
};
char (&f(std::initializer_list<char *>))[1]; // #1
char (&f(std::initializer_list<ugly>))[2];   // #2
void g() {
  // Pick #2 as #1 not viable (3->char * fails).
  static_assert(sizeof(f({"hello", 3})) == 2); // expected-warning{{not allow}}
}

} // namespace Five

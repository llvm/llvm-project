// RUN: %clang_cc1 -std=c++2a -verify %s

namespace usage_invalid {
  // FIXME: Should we diagnose a void return type?
  void voidreturn(int &param [[clang::lifetimebound]]);

  int *not_class_member() [[clang::lifetimebound]]; // expected-error {{non-member function has no implicit object parameter}}
  struct A {
    A() [[clang::lifetimebound]]; // expected-error {{cannot be applied to a constructor}}
    ~A() [[clang::lifetimebound]]; // expected-error {{cannot be applied to a destructor}}
    static int *static_class_member() [[clang::lifetimebound]]; // expected-error {{static member function has no implicit object parameter}}
    int not_function [[clang::lifetimebound]]; // expected-error {{only applies to parameters and implicit object parameters}}
    int [[clang::lifetimebound]] also_not_function; // expected-error {{cannot be applied to types}}
  };
  int *attr_with_param(int &param [[clang::lifetimebound(42)]]); // expected-error {{takes no arguments}}
}

namespace usage_ok {
  struct IntRef { int *target; };

  int &refparam(int &param [[clang::lifetimebound]]);
  int &classparam(IntRef param [[clang::lifetimebound]]);

  // Do not diagnose non-void return types; they can still be lifetime-bound.
  long long ptrintcast(int &param [[clang::lifetimebound]]) {
    return (long long)&param;
  }
  // Likewise.
  int &intptrcast(long long param [[clang::lifetimebound]]) {
    return *(int*)param;
  }

  struct A {
    A();
    A(int);
    int *class_member() [[clang::lifetimebound]];
    operator int*() [[clang::lifetimebound]];
  };

  int *p = A().class_member(); // expected-warning {{temporary whose address is used as value of local variable 'p' will be destroyed at the end of the full-expression}}
  int *q = A(); // expected-warning {{temporary whose address is used as value of local variable 'q' will be destroyed at the end of the full-expression}}
  int *r = A(1); // expected-warning {{temporary whose address is used as value of local variable 'r' will be destroyed at the end of the full-expression}}
}

# 1 "<std>" 1 3
namespace std {
  using size_t = __SIZE_TYPE__;
  struct string {
    string();
    string(const char*);

    char &operator[](size_t) const [[clang::lifetimebound]];
  };
  string operator""s(const char *, size_t);

  struct string_view {
    string_view();
    string_view(const char *p [[clang::lifetimebound]]);
    string_view(const string &s [[clang::lifetimebound]]);
  };
  string_view operator""sv(const char *, size_t);

  struct vector {
    int *data();
    size_t size();
  };

  template<typename K, typename V> struct map {};
}
# 68 "attr-lifetimebound.cpp" 2

using std::operator""s;
using std::operator""sv;

namespace p0936r0_examples {
  std::string_view s = "foo"s; // expected-warning {{temporary}}

  std::string operator+(std::string_view s1, std::string_view s2);
  void f() {
    std::string_view sv = "hi";
    std::string_view sv2 = sv + sv; // expected-warning {{temporary}}
    sv2 = sv + sv; // FIXME: can we infer that we should warn here too?
  }

  struct X { int a, b; };
  const int &f(const X &x [[clang::lifetimebound]]) { return x.a; }
  const int &r = f(X()); // expected-warning {{temporary}}

  char &c = std::string("hello my pretty long strong")[0]; // expected-warning {{temporary}}

  struct reversed_range {
    int *begin();
    int *end();
    int *p;
    std::size_t n;
  };
  template <typename R> reversed_range reversed(R &&r [[clang::lifetimebound]]) {
    return reversed_range{r.data(), r.size()};
  }

  std::vector make_vector();
  void use_reversed_range() {
    // FIXME: Don't expose the name of the internal range variable.
    for (auto x : reversed(make_vector())) {} // expected-warning {{temporary implicitly bound to local reference will be destroyed at the end of the full-expression}}
  }

  template <typename K, typename V>
  const V &findOrDefault(const std::map<K, V> &m [[clang::lifetimebound]],
                         const K &key,
                         const V &defvalue [[clang::lifetimebound]]);

  // FIXME: Maybe weaken the wording here: "local reference 'v' could bind to temporary that will be destroyed at end of full-expression"?
  std::map<std::string, std::string> m;
  const std::string &v = findOrDefault(m, "foo"s, "bar"s); // expected-warning {{temporary bound to local reference 'v'}}
}

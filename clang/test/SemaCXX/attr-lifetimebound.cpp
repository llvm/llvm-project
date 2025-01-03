// RUN: %clang_cc1 -std=c++23 -verify %s

namespace usage_invalid {
  void void_return(int &param [[clang::lifetimebound]]); // expected-error {{'lifetimebound' attribute cannot be applied to a parameter of a function that returns void; did you mean 'lifetime_capture_by(X)'}}

  int *not_class_member() [[clang::lifetimebound]]; // expected-error {{non-member function has no implicit object parameter}}
  struct A {
    A() [[clang::lifetimebound]]; // expected-error {{cannot be applied to a constructor}}
    ~A() [[clang::lifetimebound]]; // expected-error {{cannot be applied to a destructor}}
    static int *static_class_member() [[clang::lifetimebound]]; // expected-error {{static member function has no implicit object parameter}}
    int *explicit_object(this A&) [[clang::lifetimebound]]; // expected-error {{explicit object member function has no implicit object parameter}}
    int not_function [[clang::lifetimebound]]; // expected-error {{only applies to parameters and implicit object parameters}}
    int [[clang::lifetimebound]] also_not_function; // expected-error {{cannot be applied to types}}
    void void_return_member() [[clang::lifetimebound]]; // expected-error {{'lifetimebound' attribute cannot be applied to an implicit object parameter of a function that returns void; did you mean 'lifetime_capture_by(X)'}}
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

  template <class T, class R = void> R dependent_void(const T& t [[clang::lifetimebound]]);
  void dependent_void_instantiation() {
    dependent_void<int>(1); // OK: Returns void.
    int x = dependent_void<int, int>(1); // expected-warning {{temporary whose address is used as value of local variable 'x' will be destroyed at the end of the full-expression}}
    dependent_void<int, int>(1); // OK: Returns an unused value.
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

  void test_assignment() {
    p = A().class_member(); // expected-warning {{object backing the pointer p will be destroyed at the end of the full-expression}}
    p = {A().class_member()}; // expected-warning {{object backing the pointer p will be destroyed at the end of the full-expression}}
    q = A(); // expected-warning {{object backing the pointer q will be destroyed at the end of the full-expression}}
    r = A(1); // expected-warning {{object backing the pointer r will be destroyed at the end of the full-expression}}
  }

  struct FieldCheck {
    struct Set {
      int a;
    };
    struct Pair {
      const int& a;
      int b;
      Set c;
      int * d;
    };
    Pair p;  
    FieldCheck(const int& a): p(a){}
    Pair& getR() [[clang::lifetimebound]] { return p; }
    Pair* getP() [[clang::lifetimebound]] { return &p; }
    Pair* getNoLB() { return &p; }
  };
  void test_field_access() {
    int x = 0;
    const int& a = FieldCheck{x}.getR().a;
    const int& b = FieldCheck{x}.getP()->b;   // expected-warning {{temporary bound to local reference 'b' will be destroyed at the end of the full-expression}}
    const int& c = FieldCheck{x}.getP()->c.a; // expected-warning {{temporary bound to local reference 'c' will be destroyed at the end of the full-expression}}
    const int& d = FieldCheck{x}.getNoLB()->c.a;
    const int* e = FieldCheck{x}.getR().d;
  }
}

namespace std {
  using size_t = __SIZE_TYPE__;
  template<typename T>
  struct basic_string {
    basic_string();
    basic_string(const T*);

    char &operator[](size_t) const [[clang::lifetimebound]];
  };
  using string =  basic_string<char>;
  string operator""s(const char *, size_t); // expected-warning {{user-defined literal suffixes not starting with '_' are reserved}}

  template<typename T>
  struct basic_string_view {
    basic_string_view();
    basic_string_view(const T *p);
    basic_string_view(const string &s [[clang::lifetimebound]]);
  };
  using string_view = basic_string_view<char>;
  string_view operator""sv(const char *, size_t); // expected-warning {{user-defined literal suffixes not starting with '_' are reserved}}

  struct vector {
    int *data();
    size_t size();
  };

  template<typename K, typename V> struct map {};
}

using std::operator""s;
using std::operator""sv;

namespace default_args {
  using IntArray = int[];
  const int *defaultparam1(const int &def1 [[clang::lifetimebound]] = 0); // #def1
  const int &defaultparam_array([[clang::lifetimebound]] const int *p = IntArray{1, 2, 3}); // #def2
  struct A {
    A(const char *, const int &def3 [[clang::lifetimebound]] = 0); // #def3
  };
  const int &defaultparam2(const int &def4 [[clang::lifetimebound]] = 0); // #def4
  const int &defaultparam3(const int &def5 [[clang::lifetimebound]] = defaultparam2(), const int &def6 [[clang::lifetimebound]] = 0); // #def5 #def6
  std::string_view defaultparam4(std::string_view s [[clang::lifetimebound]] = std::string()); // #def7

  const int *test_default_args() {
    const int *c = defaultparam1(); // expected-warning {{temporary whose address is used as value of local variable 'c' will be destroyed at the end of the full-expression}} expected-note@#def1 {{initializing parameter 'def1' with default argument}}
    A a = A(""); // expected-warning {{temporary whose address is used as value of local variable 'a' will be destroyed at the end of the full-expression}} expected-note@#def3 {{initializing parameter 'def3' with default argument}}
    const int &s = defaultparam2(); // expected-warning {{temporary bound to local reference 's' will be destroyed at the end of the full-expression}} expected-note@#def4 {{initializing parameter 'def4' with default argument}}
    const int &t = defaultparam3(); // expected-warning {{temporary bound to local reference 't' will be destroyed at the end of the full-expression}} expected-note@#def4 {{initializing parameter 'def4' with default argument}} expected-note@#def5 {{initializing parameter 'def5' with default argument}} expected-warning {{temporary bound to local reference 't' will be destroyed at the end of the full-expression}} expected-note@#def6 {{initializing parameter 'def6' with default argument}}
    const int &u = defaultparam_array(); // expected-warning {{temporary bound to local reference 'u' will be destroyed at the end of the full-expression}} expected-note@#def2 {{initializing parameter 'p' with default argument}}
    int local;
    const int &v = defaultparam2(local); // no warning
    const int &w = defaultparam2(1); // expected-warning {{temporary bound to local reference 'w' will be destroyed at the end of the full-expression}}
    if (false) {
      return &defaultparam2();  // expected-warning {{returning address of local temporary object}}
    }
    if (false) {
      return &defaultparam2(0);  // expected-warning {{returning address of local temporary object}} expected-note@#def4 {{initializing parameter 'def4' with default argument}}
    }
    std::string_view sv = defaultparam4(); // expected-warning {{temporary whose address is used as value of local variable 'sv' will be destroyed at the end of the full-expression}} expected-note@#def7 {{initializing parameter 's' with default argument}}
    return nullptr;
  }
} // namespace default_args

namespace p0936r0_examples {
  std::string_view s = "foo"s; // expected-warning {{temporary}}

  std::string operator+(std::string_view s1, std::string_view s2);
  void f() {
    std::string_view sv = "hi";
    std::string_view sv2 = sv + sv; // expected-warning {{temporary}}
    sv2 = sv + sv; // expected-warning {{object backing the pointer}}
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

// definitions for std::move, std::forward et al.
namespace std {
inline namespace foo {

template <class T> struct remove_reference {
    typedef T type;
};
template <class T> struct remove_reference<T &> {
    typedef T type;
};
template <class T> struct remove_reference<T &&> {
    typedef T type;
};

template <class T> constexpr typename remove_reference<T>::type &&move(T &&t) {
    return static_cast<typename remove_reference<T>::type>(t);
}

template <class T>
constexpr T &&forward(typename remove_reference<T>::type &t) {
    return static_cast<T &&>(t);
}

template <class T>
constexpr T &&forward(typename remove_reference<T>::type &&t) {
    return static_cast<T &&>(t);
}

template <class T> constexpr const T &as_const(T &x) { return x; }

template <class T, bool RValueRef> struct PickRef {
    using type = typename remove_reference<T>::type &;
};
template <class T> struct PickRef<T, true> {
    using type = typename remove_reference<T>::type &&;
};

template <class T> struct is_lvalue_reference {
    static constexpr bool value = false;
};

template <class T> struct is_lvalue_reference<T &> {
    static constexpr bool value = true;
};

template <class T> struct is_const {
    static constexpr bool value = false;
};

template <class T> struct is_const<const T> {
    static constexpr bool value = true;
};

template <bool B, class T, class F> struct conditional {
    using type = T;
};

template <class T, class F> struct conditional<false, T, F> {
    using type = F;
};

template <class U, class T>
using CopyConst = typename conditional<is_const<remove_reference<U>>::value,
                                       const T, T>::type;

template <class U, class T>
using OverrideRef =
    typename conditional<is_lvalue_reference<U &&>::value,
                         typename remove_reference<T>::type &,
                         typename remove_reference<T>::type &&>::type;

template <class U, class T>
using ForwardLikeRetType = OverrideRef<U &&, CopyConst<U, T>>;

template <class U>
constexpr auto forward_like(auto &&t) -> ForwardLikeRetType<U, decltype(t)> {
    return static_cast<ForwardLikeRetType<U, decltype(t)>>(t);
}

template <class T>
auto move_if_noexcept(T &t) ->
    typename PickRef<T, noexcept(T(static_cast<T &&>(t)))>::type {
    return static_cast<
        typename PickRef<T, noexcept(T(static_cast<T &&>(t)))>::type>(t);
}

template <class T> T *addressof(T &arg) {
    return reinterpret_cast<T *>(
        &const_cast<char &>(reinterpret_cast<const volatile char &>(arg)));
}

template <class T> struct span {
  template<size_t _ArrayExtent>
	span(const T (&__arr)[_ArrayExtent]) noexcept;
};

} // namespace foo
} // namespace std

namespace move_forward_et_al_examples {
  struct S {
    S &self() [[clang::lifetimebound]] { return *this; }
  };

  S &&Move = std::move(S{}); // expected-warning {{temporary bound to local reference 'Move' will be destroyed at the end of the full-expression}}
  S MoveOk = std::move(S{});

  S &&Forward = std::forward<S &&>(S{}); // expected-warning {{temporary bound to local reference 'Forward' will be destroyed at the end of the full-expression}}
  S ForwardOk = std::forward<S &&>(S{});

  S &&ForwardLike = std::forward_like<int&&>(S{}); // expected-warning {{temporary bound to local reference 'ForwardLike' will be destroyed at the end of the full-expression}}
  S ForwardLikeOk = std::forward_like<int&&>(S{});

  const S &Const = std::as_const(S{}.self()); // expected-warning {{temporary bound to local reference 'Const' will be destroyed at the end of the full-expression}}
  const S ConstOk = std::as_const(S{}.self());

  S &&MoveIfNoExcept = std::move_if_noexcept(S{}.self()); // expected-warning {{temporary bound to local reference 'MoveIfNoExcept' will be destroyed at the end of the full-expression}}
  S MoveIfNoExceptOk = std::move_if_noexcept(S{}.self());

  S *AddressOf = std::addressof(S{}.self()); // expected-warning {{temporary whose address is used as value of local variable 'AddressOf' will be destroyed at the end of the full-expression}}
  S X;
  S *AddressOfOk = std::addressof(X);
} // namespace move_forward_et_al_examples

namespace ctor_cases {
std::basic_string_view<char> test1() {
  char abc[10];
  return abc;  // expected-warning {{address of stack memory associated with local variable}}
}

std::span<int> test2() {
  int abc[10];
  return abc; // expected-warning {{address of stack memory associated with local variable}}
}
} // namespace ctor_cases

namespace GH106372 {
class [[gsl::Owner]] Foo {};
class [[gsl::Pointer]] FooView {};

class NonAnnotatedFoo {};
class NonAnnotatedFooView {};

template <typename T>
struct StatusOr {
  template <typename U = T>
  StatusOr& operator=(U&& v [[clang::lifetimebound]]);
};

void test(StatusOr<FooView> foo1, StatusOr<NonAnnotatedFooView> foo2) {
  foo1 = Foo(); // expected-warning {{object backing foo1 will be destroyed at the end}}
  // This warning is triggered by the lifetimebound annotation, regardless of whether the class type is annotated with GSL.
  foo2 = NonAnnotatedFoo(); // expected-warning {{object backing foo2 will be destroyed at the end}}
}
} // namespace GH106372

// RUN: %clang_cc1 %s -std=c++2c -fsyntax-only -verify
namespace std {
template <typename T>
struct initializer_list {
  const T* a;
  const T* b;
  initializer_list(T* a, T* b): a{a}, b{b} {}
};
}

struct S {
  int x;
  constexpr S(int x) : x{x} {}
};

void g(int);
template <int n> constexpr int tg() { return n; }

void f1() {
  template for (auto x : {}) static_assert(false, "discarded");
  template for (constexpr auto x : {}) static_assert(false, "discarded");
  template for (auto x : {1}) g(x);
  template for (auto x : {1, 2, 3}) g(x);
  template for (constexpr auto x : {1}) g(x);
  template for (constexpr auto x : {1, 2, 3}) g(x);
  template for (constexpr auto x : {1}) tg<x>();
  template for (constexpr auto x : {1, 2, 3})
    static_assert(tg<x>());

  template for (int x : {1, 2, 3}) g(x);
  template for (S x : {1, 2, 3}) g(x.x);
  template for (constexpr S x : {1, 2, 3}) tg<x.x>();

  template for (int x : {"1", S(1), {1, 2}}) { // expected-error {{cannot initialize a variable of type 'int' with an lvalue of type 'const char[2]'}} \
                                                  expected-error {{no viable conversion from 'S' to 'int'}} \
                                                  expected-error {{excess elements in scalar initializer}} \
                                                  expected-note 3 {{in instantiation of expansion statement requested here}}
    g(x);
  }

  template for (constexpr auto x : {1, 2, 3, 4}) { // expected-note 3 {{in instantiation of expansion statement requested here}}
    static_assert(tg<x>() == 4); // expected-error 3 {{static assertion failed due to requirement 'tg<x>() == 4'}} \
                                    expected-note {{expression evaluates to '1 == 4'}} \
                                    expected-note {{expression evaluates to '2 == 4'}} \
                                    expected-note {{expression evaluates to '3 == 4'}}
  }


  template for (constexpr auto x : {1, 2}) { // expected-note 2 {{in instantiation of expansion statement requested here}}
    static_assert(false, "not discarded"); // expected-error 2 {{static assertion failed: not discarded}}
  }
}

template <typename T>
void t1() {
  template for (T x : {}) g(x);
  template for (constexpr T x : {}) g(x);
  template for (auto x : {}) g(x);
  template for (constexpr auto x : {}) g(x);
  template for (T x : {1, 2}) g(x);
  template for (T x : {T(1), T(2)}) g(x);
  template for (auto x : {T(1), T(2)}) g(x);
  template for (constexpr T x : {T(1), T(2)}) static_assert(tg<x>());
  template for (constexpr auto x : {T(1), T(2)}) static_assert(tg<x>());
}

template <typename U>
struct s1 {
  template <typename T>
  void tf() {
      template for (T x : {}) g(x);
      template for (constexpr T x : {}) g(x);
      template for (U x : {}) g(x);
      template for (constexpr U x : {}) g(x);
      template for (auto x : {}) g(x);
      template for (constexpr auto x : {}) g(x);
      template for (T x : {1, 2}) g(x);
      template for (U x : {1, 2}) g(x);
      template for (U x : {T(1), T(2)}) g(x);
      template for (T x : {U(1), U(2)}) g(x);
      template for (auto x : {T(1), T(2)}) g(x);
      template for (auto x : {U(1), T(2)}) g(x);
      template for (constexpr U x : {T(1), T(2)}) static_assert(tg<x>());
      template for (constexpr T x : {U(1), U(2)}) static_assert(tg<x>());
      template for (constexpr auto x : {T(1), U(2)}) static_assert(tg<x>());
    }
};

template <typename T>
void t2() {
  template for (T x : {}) g(x);
}

void f2() {
  t1<int>();
  t1<long>();
  s1<long>().tf<long>();
  s1<int>().tf<int>();
  s1<int>().tf<long>();
  s1<long>().tf<int>();
  t2<S>();
  t2<S[1231]>();
  t2<S***>();
}

template <__SIZE_TYPE__ size>
struct String {
  char data[size];

  template <__SIZE_TYPE__ n>
  constexpr String(const char (&str)[n]) { __builtin_memcpy(data, str, n); }

  constexpr const char* begin() const { return data; }
  constexpr const char* end() const { return data + size - 1; }
};

template <__SIZE_TYPE__ n>
String(const char (&str)[n]) -> String<n>;

constexpr int f3() {
  static constexpr String s{"abcd"};
  int count = 0;
  template for (constexpr auto x : s) count++;
  return count;
}

static_assert(f3() == 4);

void f4() {
  static constexpr String empty{""};
  static constexpr String s{"abcd"};
  template for (auto x : empty) static_assert(false, "not expanded");
  template for (constexpr auto x : s) g(x);
  template for (auto x : s) g(x);
}

struct NegativeSize {
  static constexpr const char* str = "123";
  constexpr const char* begin() const { return str + 3; }
  constexpr const char* end() const { return str; }
};

template <typename T, __SIZE_TYPE__ size>
struct Array {
  T data[size]{};
  constexpr const T* begin() const { return data; }
  constexpr const T* end() const { return data + size; }
};

void expansion_size() {
  static constexpr Array<int, 32> almost_too_big;
  template for (auto x : almost_too_big) g(x);
  template for (constexpr auto x : almost_too_big) g(x);

  static constexpr Array<int, 33> too_big;
  template for (auto x : too_big) g(x); // expected-error {{expansion size 33 exceeds maximum configured size 32}} expected-note {{use -fexpansion-limit=N to adjust this limit}}
  template for (constexpr auto x : too_big) g(x); // expected-error {{expansion size 33 exceeds maximum configured size 32}} expected-note {{use -fexpansion-limit=N to adjust this limit}}

  static constexpr String big{"1234567890123456789012345678901234567890234567890"};
  template for (auto x : big) g(x); // expected-error {{expansion size 49 exceeds maximum configured size 32}} expected-note {{use -fexpansion-limit=N to adjust this limit}}
  template for (constexpr auto x : big) g(x); // expected-error {{expansion size 49 exceeds maximum configured size 32}} expected-note {{use -fexpansion-limit=N to adjust this limit}}

  static constexpr NegativeSize n;
  template for (auto x : n) g(x); // expected-error {{expansion size must not be negative (was -3)}}
  template for (constexpr auto x : n) g(x); // expected-error {{expansion size must not be negative (was -3)}}

  template for (auto x : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32}) g(x);
  template for (constexpr auto x : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32}) g(x);

  template for (auto x : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, // expected-error {{expansion size 33 exceeds maximum configured size 32}} expected-note {{use -fexpansion-limit=N to adjust this limit}}
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33}) g(x);
  template for (constexpr auto x : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, // expected-error {{expansion size 33 exceeds maximum configured size 32}} expected-note {{use -fexpansion-limit=N to adjust this limit}}
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33}) g(x);
}

struct NotInt {
  struct iterator {};
  constexpr iterator begin() const { return {}; }
  constexpr iterator end() const { return {}; }
};

void not_int() {
  static constexpr NotInt ni;
  template for (auto x : ni) g(x); // expected-error {{invalid operands to binary expression}}
}

static constexpr Array<int, 3> integers{1, 2, 3};

constexpr int friend_func();

struct Private {
  friend constexpr int friend_func();

private:
  constexpr const int* begin() const { return integers.begin(); } // expected-note 3 {{declared private here}}
  constexpr const int* end() const { return integers.end(); } // expected-note 1 {{declared private here}}

public:
  static constexpr int member_func() {
    int sum = 0;
    static constexpr Private p1;
    template for (auto x : p1) sum += x;
    return sum;
  }
};

struct Protected {
  friend constexpr int friend_func();

protected:
  constexpr const int* begin() const { return integers.begin(); } // expected-note 3 {{declared protected here}}
  constexpr const int* end() const { return integers.end(); } // expected-note 1 {{declared protected here}}

public:
  static constexpr int member_func() {
    int sum = 0;
    static constexpr Protected p1;
    template for (auto x : p1) sum += x;
    return sum;
  }
};

void access_control() {
  static constexpr Private p1;
  template for (auto x : p1) g(x); // expected-error 3 {{'begin' is a private member of 'Private'}} expected-error 1 {{'end' is a private member of 'Private'}}

  static constexpr Protected p2;
  template for (auto x : p2) g(x); // expected-error 3 {{'begin' is a protected member of 'Protected'}} expected-error 1 {{'end' is a protected member of 'Protected'}}
}

constexpr int friend_func() {
  int sum = 0;
  static constexpr Private p1;
  template for (auto x : p1) sum += x;

  static constexpr Protected p2;
  template for (auto x : p2) sum += x;
  return sum;
}

static_assert(friend_func() == 12);
static_assert(Private::member_func() == 6);
static_assert(Protected::member_func() == 6);

struct SizeNotICE {
  struct iterator {
    friend constexpr iterator operator+(iterator a, __PTRDIFF_TYPE__) { return a; }
    int constexpr operator*() const { return 7; }

    // NOT constexpr!
    friend int operator-(iterator, iterator) { return 7; } // expected-note {{declared here}}
    friend int operator!=(iterator, iterator) { return 7; }
  };
  constexpr iterator begin() const { return {}; }
  constexpr iterator end() const { return {}; }
};

struct PlusMissing {
  struct iterator {
    int constexpr operator*() const { return 7; }
  };
  constexpr iterator begin() const { return {}; }
  constexpr iterator end() const { return {}; }
};

struct DerefMissing {
  struct iterator {
    friend constexpr iterator operator+(iterator a, __PTRDIFF_TYPE__) { return a; }
  };
  constexpr iterator begin() const { return {}; }
  constexpr iterator end() const { return {}; }
};

void missing_funcs() {
  static constexpr SizeNotICE s1;
  static constexpr PlusMissing s2;
  static constexpr DerefMissing s3;

  // TODO: This message should start complaining about '!=' once we support the
  // proper way of computing the size.
  template for (auto x : s1) g(x); // expected-error {{expansion size is not a constant expression}} \
                                      expected-note {{non-constexpr function 'operator-' cannot be used in a constant expression}}

  template for (auto x : s2) g(x); // expected-error {{invalid operands to binary expression}}
  template for (auto x : s3) g(x); // expected-error {{indirection requires pointer operand ('iterator' invalid)}}
}

namespace adl {
struct ADL {

};

constexpr const int* begin(const ADL&) { return integers.begin(); }
constexpr const int* end(const ADL&) { return integers.end(); }
}

namespace adl_error {
struct ADLError1 {
  constexpr const int* begin() const { return integers.begin(); } // expected-note {{member is not a candidate because range type 'const adl_error::ADLError1' has no 'end' member}}
};

struct ADLError2 {
  constexpr const int* end() const { return integers.end(); } // expected-note {{member is not a candidate because range type 'const adl_error::ADLError2' has no 'begin' member}}
};

constexpr const int* begin(const ADLError2&) { return integers.begin(); } // expected-note {{candidate function not viable: no known conversion from 'const adl_error::ADLError1' to 'const ADLError2' for 1st argument}}
constexpr const int* end(const ADLError1&) { return integers.end(); } // expected-note {{candidate function not viable: no known conversion from 'const adl_error::ADLError2' to 'const ADLError1' for 1st argument}}
}

namespace adl_both {
static constexpr Array<int, 5> integers2{1, 2, 3, 4, 5};
struct ADLBoth {
  // Test that member begin/end are preferred over ADl begin/end. These return
  // pointers to a different array.
  constexpr const int* begin() const { return integers2.begin(); }
  constexpr const int* end() const { return integers2.end(); }
};

constexpr const int* begin(const ADLBoth&) { return integers.begin(); }
constexpr const int* end(const ADLBoth&) { return integers.end(); }
}

constexpr int adl_begin_end() {
  static constexpr adl::ADL a;
  int sum = 0;
  template for (auto x : a) sum += x;
  template for (constexpr auto x : a) sum += x;
  return sum;
}

static_assert(adl_begin_end() == 12);

void adl_mixed_error() {
  static constexpr adl_error::ADLError1 a1;
  static constexpr adl_error::ADLError2 a2;
  template for (auto x : a1) g(x); // expected-error {{invalid range expression of type 'const adl_error::ADLError1'; no viable 'begin' function available}}
  template for (auto x : a2) g(x); // expected-error {{invalid range expression of type 'const adl_error::ADLError2'; no viable 'end' function available}}
}

constexpr int adl_both_test() {
  static constexpr adl_both::ADLBoth a;
  int sum = 0;
  template for (auto x : a) sum += x;
  return sum;
}

static_assert(adl_both_test() == 15);

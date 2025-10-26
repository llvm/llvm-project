// RUN: %clang_cc1 %s -std=c++2c -fsyntax-only -fdeclspec -verify
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

void g(int); // #g
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

template <String s>
constexpr int tf3() {
  int count = 0;
  template for (constexpr auto x : s) count++;
  return count;
}

static_assert(f3() == 4);
static_assert(tf3<"1">() == 1);
static_assert(tf3<"12">() == 2);
static_assert(tf3<"123">() == 3);
static_assert(tf3<"1234">() == 4);

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
  constexpr const int* begin() const { return integers.begin(); } // expected-note 2 {{declared private here}}
  constexpr const int* end() const { return integers.end(); } // expected-note 2 {{declared private here}}

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
  constexpr const int* begin() const { return integers.begin(); } // expected-note 2 {{declared protected here}}
  constexpr const int* end() const { return integers.end(); } // expected-note 2 {{declared protected here}}

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
  template for (auto x : p1) g(x); // expected-error 2 {{'begin' is a private member of 'Private'}} expected-error 2 {{'end' is a private member of 'Private'}}

  static constexpr Protected p2;
  template for (auto x : p2) g(x); // expected-error 2 {{'begin' is a protected member of 'Protected'}} expected-error 2 {{'end' is a protected member of 'Protected'}}
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

struct A {};
struct B { int x = 1; };
struct C { int a = 1, b = 2, c = 3; };
struct D {
  int a = 1;
  int* b = nullptr;
  const char* c = "3";
};

struct Nested {
  A a;
  B b;
  C c;
};

struct PrivateDestructurable {
  friend void destructurable_friend();
private:
  int a, b; // expected-note 4 {{declared private here}}
};

struct ProtectedDestructurable {
  friend void destructurable_friend();
protected:
  int a, b; // expected-note 4 {{declared protected here}}
};

void destructuring() {
  static constexpr A a;
  static constexpr B b;
  static constexpr C c;
  static constexpr D d;

  template for (auto x : a) static_assert(false, "not expanded");
  template for (constexpr auto x : a) static_assert(false, "not expanded");

  template for (auto x : b) g(x);
  template for (constexpr auto x : b) g(x);

  template for (auto x : c) g(x);
  template for (constexpr auto x : c) g(x);

  template for (auto x : d) { // expected-note 2 {{in instantiation of expansion statement requested here}}
    // expected-note@#g {{candidate function not viable: no known conversion from 'int *' to 'int' for 1st argument}}
    // expected-note@#g {{candidate function not viable: no known conversion from 'const char *' to 'int' for 1st argument}}
    g(x); // expected-error 2 {{no matching function for call to 'g'}}

  }

  template for (constexpr auto x : d) { // expected-note 2 {{in instantiation of expansion statement requested here}}
    // expected-note@#g {{candidate function not viable: no known conversion from 'int *const' to 'int' for 1st argument}}
    // expected-note@#g {{candidate function not viable: no known conversion from 'const char *const' to 'int' for 1st argument}}
    g(x); // expected-error 2 {{no matching function for call to 'g'}}
  }
}

constexpr int array() {
  static constexpr int x[4]{1, 2, 3, 4};
  int sum = 0;
  template for (auto y : x) sum += y;
  template for (constexpr auto y : x) sum += y;
  return sum;
}

static_assert(array() == 20);

void array_too_big() {
  int ok[32];
  int too_big[33];

  template for (auto x : ok) {}
  template for (auto x : too_big) {} // expected-error {{expansion size 33 exceeds maximum configured size 32}} \
                                        expected-note {{use -fexpansion-limit=N to adjust this limit}}
}

template <auto v>
constexpr int destructure() {
  int sum = 0;
  template for (auto x : v) sum += x;
  template for (constexpr auto x : v) sum += x;
  return sum;
}

static_assert(destructure<B{10}>() == 20);
static_assert(destructure<C{}>() == 12);
static_assert(destructure<C{3, 4, 5}>() == 24);

constexpr int nested() {
  static constexpr Nested n;
  int sum = 0;
  template for (constexpr auto x : n) {
    static constexpr auto val = x;
    template for (auto y : val) {
      sum += y;
    }
  }
  template for (constexpr auto x : n) {
    static constexpr auto val = x;
    template for (constexpr auto y : val) {
      sum += y;
    }
  }
  return sum;
}

static_assert(nested() == 14);

void access_control_destructurable() {
  template for (auto x : PrivateDestructurable()) {} // expected-error 2 {{cannot bind private member 'a' of 'PrivateDestructurable'}} \
                                                        expected-error 2 {{cannot bind private member 'b' of 'PrivateDestructurable'}}

  template for (auto x : ProtectedDestructurable()) {} // expected-error 2 {{cannot bind protected member 'a' of 'ProtectedDestructurable'}} \
                                                          expected-error 2 {{cannot bind protected member 'b' of 'ProtectedDestructurable'}}
}

void destructurable_friend() {
  template for (auto x : PrivateDestructurable()) {}
  template for (auto x : ProtectedDestructurable()) {}
}

struct Placeholder {
  A get_value() const { return {}; }
  __declspec(property(get = get_value)) A a;
};

void placeholder() {
  template for (auto x: Placeholder().a) {}
}

union Union { int a; long b;};

struct MemberPtr {
  void f() {}
};

void overload_set(int); // expected-note 2 {{possible target for call}}
void overload_set(long); // expected-note 2 {{possible target for call}}

void invalid_types() {
  template for (auto x : void()) {} // expected-error {{cannot expand expression of type 'void'}}
  template for (auto x : 1) {} // expected-error {{cannot expand expression of type 'int'}}
  template for (auto x : 1.f) {} // expected-error {{cannot expand expression of type 'float'}}
  template for (auto x : 'c') {} // expected-error {{cannot expand expression of type 'char'}}
  template for (auto x : invalid_types) {} // expected-error {{cannot expand expression of type 'void ()'}}
  template for (auto x : &invalid_types) {} // expected-error {{cannot expand expression of type 'void (*)()'}}
  template for (auto x : &MemberPtr::f) {} // expected-error {{cannot expand expression of type 'void (MemberPtr::*)()'}}
  template for (auto x : overload_set) {} // expected-error{{reference to overloaded function could not be resolved; did you mean to call it?}}
  template for (auto x : &overload_set) {} // expected-error{{reference to overloaded function could not be resolved; did you mean to call it?}}
  template for (auto x : nullptr) {} // expected-error {{cannot expand expression of type 'std::nullptr_t'}}
  template for (auto x : __builtin_strlen) {} // expected-error {{builtin functions must be directly called}}
  template for (auto x : Union()) {} // expected-error {{cannot expand expression of type 'Union'}}
  template for (auto x : (char*)nullptr) {} // expected-error {{cannot expand expression of type 'char *'}}
  template for (auto x : []{}) {} // expected-error {{cannot expand lambda closure type}}
  template for (auto x : [x=3]{}) {} // expected-error {{cannot expand lambda closure type}}
}

struct BeginOnly {
  int x{1};
  constexpr const int* begin() const { return nullptr; }
};

struct EndOnly {
  int x{2};
  constexpr const int* end() const { return nullptr; }
};

namespace adl1 {
struct BeginOnly {
  int x{3};
};
constexpr const int* begin(const BeginOnly&) { return nullptr; }
}

namespace adl2 {
struct EndOnly {
  int x{4};
};
constexpr const int* end(const EndOnly&) { return nullptr; }
}

constexpr int unpaired_begin_end() {
  static constexpr BeginOnly b1;
  static constexpr EndOnly e1;
  static constexpr adl1::BeginOnly b2;
  static constexpr adl2::EndOnly e2;
  int sum = 0;

  template for (auto x : b1) sum += x;
  template for (auto x : e1) sum += x;

  template for (auto x : b2) sum += x;
  template for (auto x : e2) sum += x;

  return sum;
}

static_assert(unpaired_begin_end() == 10);

// Examples taken from [stmt.expand].
namespace stmt_expand_examples {
consteval int f(auto const&... Containers) {
  int result = 0;
  template for (auto const& c : {Containers...}) {      // OK, enumerating expansion statement
    result += c[0];
  }
  return result;
}
constexpr int c1[] = {1, 2, 3};
constexpr int c2[] = {4, 3, 2, 1};
static_assert(f(c1, c2) == 5);

// TODO: This entire example should work without issuing any diagnostics once
// we have full support for references to constexpr variables (P2686).
consteval int f() {
  constexpr Array<int, 3> arr {1, 2, 3}; // expected-note{{add 'static' to give it a constant address}}

  int result = 0;

  // expected-error@#invalid-ref {{constexpr variable '__range1' must be initialized by a constant expression}}
  // expected-error@#invalid-ref {{constexpr variable '__begin1' must be initialized by a constant expression}}
  // expected-error@#invalid-ref {{constexpr variable '__end1' must be initialized by a constant expression}}
  // expected-error@#invalid-ref {{expansion size is not a constant expression}}
  // expected-note@#invalid-ref 2 {{member call on variable '__range1' whose value is not known}}
  // expected-note@#invalid-ref 1 {{initializer of '__end1' is not a constant expression}}
  // expected-note@#invalid-ref 3 {{declared here}}
  // expected-note@#invalid-ref {{reference to 'arr' is not a constant expression}}
  template for (constexpr int s : arr) { // #invalid-ref                // OK, iterating expansion statement
    result += sizeof(char[s]);
  }
  return result;
}
static_assert(f() == 6); // expected-error {{static assertion failed due to requirement 'f() == 6'}} expected-note {{expression evaluates to '0 == 6'}}

struct S {
  int i;
  short s;
};

consteval long f(S s) {
  long result = 0;
  template for (auto x : s) {                           // OK, destructuring expansion statement
    result += sizeof(x);
  }
  return result;
}
static_assert(f(S{}) == sizeof(int) + sizeof(short));
}

void not_constant_expression() {
  template for (constexpr auto x : B()) { // expected-error {{constexpr variable '[__u0]' must be initialized by a constant expression}} \
                                             expected-note {{reference to temporary is not a constant expression}} \
                                             expected-note {{temporary created here}} \
                                             expected-error {{constexpr variable 'x' must be initialized by a constant expression}} \
                                             expected-note {{in instantiation of expansion statement requested here}} \
                                             expected-note {{read of variable '[__u0]' whose value is not known}} \
                                             expected-note {{declared here}}
    g(x);
  }
}

constexpr int references_enumerating() {
  int x = 1, y = 2, z = 3;
  template for (auto& x : {x, y, z}) { ++x; }
  template for (auto&& x : {x, y, z}) { ++x; }
  return x + y + z;
}

static_assert(references_enumerating() == 12);

constexpr int references_destructuring() {
  C c;
  template for (auto& x : c) { ++x; }
  template for (auto&& x : c) { ++x; }
  return c.a + c.b + c.c;
}

static_assert(references_destructuring() == 12);

constexpr int break_continue() {
  int sum = 0;
  template for (auto x : {1, 2}) {
    break;
    sum += x;
  }

  template for (auto x : {3, 4}) {
    continue;
    sum += x;
  }

  template for (auto x : {5, 6}) {
    if (x == 6) break;
    sum += x;
  }

  template for (auto x : {7, 8, 9}) {
    if (x == 8) continue;
    sum += x;
  }

  return sum;
}

static_assert(break_continue() == 21);

constexpr int break_continue_nested() {
  int sum = 0;

  template for (auto x : {1, 2}) {
    template for (auto y : {3, 4}) {
      if (x == 2) break;
      sum += y;
    }
    sum += x;
  }

  template for (auto x : {5, 6}) {
    template for (auto y : {7, 8}) {
      if (x == 6) continue;
      sum += y;
    }
    sum += x;
  }

  return sum;
}

static_assert(break_continue_nested() == 36);

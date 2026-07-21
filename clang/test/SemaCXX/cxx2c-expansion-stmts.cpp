// RUN: %clang_cc1 %s -std=c++2c -fsyntax-only -fdeclspec -fblocks -Wno-vla-cxx-extension -fconstexpr-steps=10000 -verify=expected,old-interp
// RUN: %clang_cc1 %s -std=c++2c -fsyntax-only -fdeclspec -fblocks -Wno-vla-cxx-extension -fconstexpr-steps=10000 -verify=expected,new-interp -fexperimental-new-constant-interpreter
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

// Note: Remove this test once we do support them.
int iterating_expansion_stmts_unsupported() {
  static constexpr String s{"abcd"};
  int count = 0;
  template for (constexpr auto x : s) count++; // expected-error {{iterating expansion statements are not yet supported}}
  return count;
}

template <typename T>
int iterating_expansion_stmts_unsupported_dependent() {
  static constexpr String s{"abcd"};
  int count = 0;
  template for (auto x : T(s)) count++; // expected-error {{iterating expansion statements are not yet supported}}
  return count;
}

void iterating_expansion_stmts_unsupported_dependent_instantiate() {
  iterating_expansion_stmts_unsupported_dependent<String<5>>(); // expected-note {{in instantiation of}}
}

#if 0 // Disabled until we support iterating expansion statements.
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

void negative_size() {
  static constexpr NegativeSize n;
  template for (auto x : n) g(x); // expected-error {{expansion statement size is not a constant expression}} \
                                     old-interp-note {{constexpr evaluation hit maximum step limit}} \
                                     new-interp-note {{cannot refer to element 5 of array of 4 elements in a constant expression}} \
                                     expected-note {{in call to}}
  template for (constexpr auto x : n) g(x); // expected-error {{expansion statement size is not a constant expression}} \
                                               old-interp-note {{constexpr evaluation hit maximum step limit}} \
                                               new-interp-note {{cannot refer to element 5 of array of 4 elements in a constant expression}} \
                                               expected-note {{in call to}}
}

template <typename T, __SIZE_TYPE__ size>
struct Array {
  T data[size]{};
  constexpr const T* begin() const { return data; }
  constexpr const T* end() const { return data + size; }
};

struct NotInt {
  struct iterator {};
  constexpr iterator begin() const { return {}; }
  constexpr iterator end() const { return {}; }
};

void not_int() {
  static constexpr NotInt ni;
  template for (auto x : ni) g(x); // expected-error {{invalid operands to binary expression}} \
                                      expected-note {{while attempting to construct 'begin - begin' with iterator type 'iterator'}}
}

static constexpr Array<int, 3> integers{1, 2, 3};

constexpr int friend_func();

struct Private {
  friend constexpr int friend_func();

private:
  constexpr const int* begin() const { return integers.begin(); } // expected-note 3 {{declared private here}}
  constexpr const int* end() const { return integers.end(); } // expected-note 3 {{declared private here}}

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
  constexpr const int* end() const { return integers.end(); } // expected-note 3 {{declared protected here}}

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
  template for (auto x : p1) g(x); // expected-error 3 {{'begin' is a private member of 'Private'}} expected-error 3 {{'end' is a private member of 'Private'}}

  static constexpr Protected p2;
  template for (auto x : p2) g(x); // expected-error 3 {{'begin' is a protected member of 'Protected'}} expected-error 3 {{'end' is a protected member of 'Protected'}}
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
    friend constexpr __PTRDIFF_TYPE__ operator-(iterator, iterator) { return 7; }
    constexpr iterator operator++() { return *this; }
    int constexpr operator*() const { return 7; }

    // NOT constexpr!
    friend int operator!=(iterator, iterator) { return 7; } // expected-note {{declared here}}
  };
  constexpr iterator begin() const { return {}; }
  constexpr iterator end() const { return {}; }
};

struct PlusMissing {
  struct iterator {
    friend constexpr __PTRDIFF_TYPE__ operator-(iterator, iterator) { return 7; }
    constexpr iterator operator++() { return *this; }
    int constexpr operator*() const { return 7; }
  };
  constexpr iterator begin() const { return {}; }
  constexpr iterator end() const { return {}; }
};

struct DerefMissing {
  struct iterator {
    friend constexpr __PTRDIFF_TYPE__ operator-(iterator, iterator) { return 7; }
    friend constexpr iterator operator+(iterator a, __PTRDIFF_TYPE__) { return a; }
  };
  constexpr iterator begin() const { return {}; }
  constexpr iterator end() const { return {}; }
};

struct MinusMissing {
  struct iterator {};
  constexpr iterator begin() const { return {}; }
  constexpr iterator end() const { return {}; }
};

void missing_funcs() {
  static constexpr SizeNotICE s1;
  static constexpr PlusMissing s2;
  static constexpr DerefMissing s3;
  static constexpr MinusMissing s4;

  template for (auto x : s1) g(x); // expected-error {{expansion statement size is not a constant expression}} \
                                      expected-note {{non-constexpr function 'operator!=' cannot be used in a constant expression}} \
                                      expected-note {{in call to}}

  template for (auto x : s2) g(x); // expected-error {{invalid operands to binary expression}}
  template for (auto x : s3) g(x); // expected-error {{indirection requires pointer operand ('iterator' invalid)}}
  template for (auto x : s4) g(x); // expected-error {{invalid operands to binary expression ('iterator' and 'iterator')}} \
                                      expected-note {{while attempting to construct 'begin - begin' with iterator type 'iterator'}}
}

namespace adl {
struct ADL {

};

constexpr const int* begin(const ADL&) { return integers.begin(); }
constexpr const int* end(const ADL&) { return integers.end(); }
}

namespace adl_error {
struct ADLError1 {
  constexpr const int* begin() const { return integers.begin(); }
};

struct ADLError2 {
  constexpr const int* end() const { return integers.end(); }
};

constexpr const int* begin(const ADLError2&) { return integers.begin(); }
constexpr const int* end(const ADLError1&) { return integers.end(); }
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

void adl_mixed() {
  static constexpr adl_error::ADLError1 a1;
  static constexpr adl_error::ADLError2 a2;

  // These are actually destructuring because there is no
  // valid begin/end pair.
  template for (auto x : a1) g(x);
  template for (auto x : a2) g(x);
}

constexpr int adl_both_test() {
  static constexpr adl_both::ADLBoth a;
  int sum = 0;
  template for (auto x : a) sum += x;
  return sum;
}

static_assert(adl_both_test() == 15);
#endif // 0

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

  struct Derived : ProtectedDestructurable {
    void f() {
      template for (auto x : *this) {}
    }
  };
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
  template for (auto x : void()) {} // expected-error {{cannot expand expression of incomplete type 'void'}}
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
  template for (auto x : [](auto){}) {} // expected-error {{cannot expand lambda closure type}}
  template for (auto x : []<typename>(){}) {} // expected-error {{cannot expand lambda closure type}}
}

constexpr int string_literals() {
  int i = 0;
  template for (auto x : "1234") i += int(x);
  template for (auto x : L"1234") i += int(x);
  template for (auto x : u"1234") i += int(x);
  template for (auto x : U"1234") i += int(x);
  template for (auto x : R"(1234)") i += int(x);
  return i;
}

static_assert(string_literals() == 5 * (int('1') + int('2') + int('3') + int('4')));

struct Bitfields {
  int x : 5 = 1;
  int y : 5 = 2;
  int _ : 5 = 3;
};

constexpr int bitfields() {
  int i = 0;
  template for (auto x : Bitfields()) i += x;
  return i;
}

static_assert(bitfields() == 6);

struct EnumMember {
  enum {
    x, y, z
  };

  enum class Foo {
    a, b, c
  };

  using enum Foo;
};

constexpr int enum_member() {
  int i = 0;
  template for (auto x : EnumMember()) i += x;
  return i;
}

static_assert(enum_member() == 0);

struct Mutable {
  mutable int x{};
};

constexpr int mutable_member() {
  const Mutable m;
  template for (auto& i : m) i = 42;
  return m.x;
}

static_assert(mutable_member() == 42);

constexpr int vector_type() {
  __attribute__((vector_size(sizeof(int) * 4))) int vec{1, 2, 3, 4};
  int i = 0;
  template for (auto x : vec) i += x;
  return i;
}

static_assert(vector_type() == 10);

constexpr int complex() {
  _Complex int c{1, 2};
  int i = 0;
  template for (auto x : c) i += x;
  return i;
}

static_assert(complex() == 3);

#if 0 // Disabled until we support iterating expansion statements.
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

namespace adl3 {
struct BeginOnlyDeleted {
  int x{4};
};
constexpr const int* begin(const BeginOnlyDeleted&) = delete;
}

namespace adl4 {
struct EndOnlyDeleted {
  int x{4};
};
constexpr const int* end(const EndOnlyDeleted&) = delete;
}

namespace adl5 {
struct BothDeleted {
  int x{4};
};
constexpr const int* begin(const BothDeleted&) = delete; // expected-note {{candidate function has been explicitly deleted}}
constexpr const int* end(const BothDeleted&) = delete;
}

namespace adl6 {
struct BeginNotViable {
  int x{4};
};
constexpr const int* begin(int) { return nullptr; }
}

namespace adl7 {
struct EndNotViable {
  int x{4};
};
constexpr const int* end(int) { return nullptr; }
}

namespace adl8 {
struct BothNotViable {
  int x{4};
};
constexpr const int* begin(int) { return nullptr; }
constexpr const int* end(int) { return nullptr; }
}

namespace adl9 {
struct BeginDeleted {
  int x{4};
};
constexpr const int* begin(const BeginDeleted&) = delete; // expected-note {{candidate function has been explicitly deleted}}
constexpr const int* end(const BeginDeleted&) { return nullptr; }
}

namespace adl10 {
struct EndDeleted {
  int x{4};
};
constexpr const int* begin(const EndDeleted&) { return nullptr; }
constexpr const int* end(const EndDeleted&) = delete; // expected-note {{candidate function has been explicitly deleted}}
}

void unpaired_begin_end() {
  static constexpr adl1::BeginOnly begin_only;
  static constexpr adl2::EndOnly end_only;
  static constexpr adl3::BeginOnlyDeleted begin_only_deleted;
  static constexpr adl4::EndOnlyDeleted end_only_deleted;
  static constexpr adl5::BothDeleted both_deleted;
  static constexpr adl6::BeginNotViable begin_not_viable;
  static constexpr adl7::EndNotViable end_not_viable;
  static constexpr adl8::BothNotViable both_not_viable;
  static constexpr adl9::BeginDeleted begin_deleted;
  static constexpr adl10::EndDeleted end_deleted;

  // Ok, these are destructuring because there is no valid pair.
  template for (auto x : begin_only) {}
  template for (auto x : begin_only_deleted) {}
  template for (auto x : begin_not_viable) {}
  template for (auto x : end_only) {}
  template for (auto x : end_only_deleted) {}
  template for (auto x : end_not_viable) {}

  // This is also ok because overload resolution fails.
  template for (auto x : both_not_viable) {}

  // These are invalid because overload resolution succeeds (even though
  // there is no usable begin() and/or end()).
  template for (auto x : both_deleted) {} // expected-error {{call to deleted function 'begin'}} \
                                             expected-note {{when looking up 'begin' function for range expression of type 'const adl5::BothDeleted'}}~

  template for (auto x : begin_deleted) {} // expected-error {{call to deleted function 'begin'}} \
                                              expected-note {{when looking up 'begin' function for range expression of type 'const adl9::BeginDeleted'}}

  template for (auto x : end_deleted) {} // expected-error {{call to deleted function 'end'}} \
                                            expected-note {{when looking up 'end' function for range expression of type 'const adl10::EndDeleted'}}
}
#endif // 0

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

#if 0 // Disabled until we support iterating expansion statements.
// TODO: This entire example should work without issuing any diagnostics once
// we have full support for references to constexpr variables (P2686).
consteval int f() {
  constexpr Array<int, 3> arr {1, 2, 3}; // expected-note{{add 'static' to give it a constant address}} \
                                            new-interp-note 5 {{add 'static' to give it a constant address}}

  int result = 0;

  // expected-error@#invalid-ref {{constexpr variable '__range1' must be initialized by a constant expression}}
  // expected-error@#invalid-ref {{constexpr variable '__begin1' must be initialized by a constant expression}}
  // expected-error@#invalid-ref {{constexpr variable '__end1' must be initialized by a constant expression}}
  // expected-note@#invalid-ref {{reference to 'arr' is not a constant expression}}
  // old-interp-error@#invalid-ref {{expansion statement size is not a constant expression}}
  // old-interp-note@#invalid-ref {{in call to}}
  // old-interp-note@#invalid-ref 3 {{member call on variable '__range1' whose value is not known}}
  // old-interp-note@#invalid-ref 3 {{declared here}}
  // new-interp-error@#invalid-ref 3 {{constexpr variable '__iter1' must be initialized by a constant expression}}
  // new-interp-note@#invalid-ref 5 {{pointer to subobject of 'arr' is not a constant expression}}
  // new-interp-note@#invalid-ref 3 {{in instantiation of expansion statement}}
  template for (constexpr int s : arr) { // #invalid-ref                // OK, iterating expansion statement
    result += sizeof(char[s]);
  }
  return result;
}
static_assert(f() == 6); // old-interp-error {{static assertion failed due to requirement 'f() == 6'}} old-interp-note {{expression evaluates to '0 == 6'}}
#endif // 0

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
                                             old-interp-note {{read of variable '[__u0]' whose value is not known}} \
                                             old-interp-note {{declared here}} \
                                             new-interp-note {{temporary created here}} \
                                             new-interp-note {{read of temporary is not allowed in a constant expression outside the expression that created the temporary}}
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

template <typename ...Ts>
void unexpanded_pack_bad(Ts ...ts) {
  template for (auto x : ts) {} // expected-error {{expression contains unexpanded parameter pack 'ts'}}
  template for (Ts x : {1, 2}) {} // expected-error {{declaration type contains unexpanded parameter pack 'Ts'}}
  template for (auto x : {ts}) {} // expected-error {{initializer contains unexpanded parameter pack}} \
  // expected-note {{in instantiation of expansion statement requested here}}
}

struct E { int x, y; constexpr E(int x, int y) : x{x}, y{y} {}};

template <typename ...Es>
constexpr int unexpanded_pack_good(Es ...es) {
  int sum = 0;
  ([&] {
    template for (auto x : es) sum += x;
    template for (Es e : {{5, 6}, {7, 8}}) sum += e.x + e.y;
  }(), ...);
  return sum;
}

static_assert(unexpanded_pack_good(E{1, 2}, E{3, 4}) == 62);

// Ensure that the expansion-initializer is evaluated even if it expands
// to nothing.
//
// This is related to CWG 3048. Note that we currently still model this as
// a DecompositionDecl w/ zero bindings.
constexpr bool empty_side_effect() {
  struct A {
    constexpr A(bool& b) {
      b = true;
    }
  };

  bool constructed = false;
  template for (auto x : A(constructed)) static_assert(false);
  return constructed;
}

static_assert(empty_side_effect());

namespace apply_lifetime_extension {
struct T {
  int& x;
  constexpr T(int& x) noexcept : x(x) {}
  constexpr ~T() noexcept { x = 42; }
};

constexpr const T& f(const T& t) noexcept { return t; }
constexpr T g(int& x) noexcept { return T(x); }

// CWG 3043:
//
// Lifetime extension only applies to destructuring expansion statements
// (enumerating statements don't have a range variable, and the range variable
// of iterating statements is constexpr).
constexpr int lifetime_extension() {
  int x = 5;
  int sum  = 0;
  template for (auto e : f(g(x))) {
    sum += x;
  }
  return sum + x;
}

template <typename T>
constexpr int lifetime_extension_instantiate_expansions() {
  int x = 5;
  int sum  = 0;
  template for (T e : f(g(x))) {
    sum += x;
  }
  return sum + x;
}

template <typename T>
constexpr int lifetime_extension_dependent_expansion_stmt() {
  int x = 5;
  int sum  = 0;
  template for (int e : f(g((T&)x))) {
    sum += x;
  }
  return sum + x;
}

template <typename U>
struct foo {
  template <typename T>
  constexpr int lifetime_extension_multiple_instantiations() {
      int x = 5;
      int sum  = 0;
      template for (T e : f(g((U&)x))) {
        sum += x;
      }
      return sum + x;
  }
};

static_assert(lifetime_extension() == 47);
static_assert(lifetime_extension_instantiate_expansions<int>() == 47);
static_assert(lifetime_extension_dependent_expansion_stmt<int>() == 47);
static_assert(foo<int>().lifetime_extension_multiple_instantiations<int>() == 47);
}

template <typename... Ts>
constexpr int return_from_expansion(Ts... ts) {
  template for (int i : {1, 2, 3}) {
    return (ts + ...);
  }
  __builtin_unreachable();
}

static_assert(return_from_expansion(4, 5, 6) == 15);

void not_constexpr();

constexpr int empty_expansion_consteval() {
  template for (auto _ : {}) {
    not_constexpr();
  }
  return 3;
}

static_assert(empty_expansion_consteval() == 3);

void nested_empty_expansion() {
  template for (auto x1 : {})
    template for (auto x2 : {1})
      static_assert(false);

  template for (auto x1 : {1})
    template for (auto x2 : {})
      template for (auto x3 : {1})
        static_assert(false);

  template for (auto x1 : {})
    template for (auto x2 : {})
      template for (auto x3 : {})
        template for (auto x4 : {1})
          static_assert(false);

  template for (auto x1 : {})
    template for (auto x2 : {1})
      template for (auto x3 : {})
        template for (auto x4 : {1})
          static_assert(false);

  template for (auto x1 : {})
    template for (auto x2 : {1})
      template for (auto x4 : {1})
        static_assert(false);
}

struct Empty {};

template <typename T>
void nested_empty_expansion_dependent() {
  template for (auto x1 : T())
    template for (auto x2 : {1})
      static_assert(false);

  template for (auto x1 : {1})
    template for (auto x2 : T())
      template for (auto x3 : {1})
        static_assert(false);

  template for (auto x1 : T())
    template for (auto x2 : T())
      template for (auto x3 : T())
        template for (auto x4 : {1})
          static_assert(false);

  template for (auto x1 : T())
    template for (auto x2 : {1})
      template for (auto x3 : T())
        template for (auto x4 : {1})
          static_assert(false);

  template for (auto x1 : T())
    template for (auto x2 : {1})
      template for (auto x4 : {1})
        static_assert(false);
}

void nested_empty_expansion_dependent_instantiate() {
  nested_empty_expansion_dependent<Empty>();
}

// Destructuring expansion statements using tuple_size/tuple_element/get.
namespace std {
template <typename>
struct tuple_size;

template <__SIZE_TYPE__, typename>
struct tuple_element; // expected-note {{template is declared here}}

namespace get_decomposition {
struct MemberGet {
  int x[6]{};

  template <__SIZE_TYPE__ I>
  constexpr int& get() { return x[I * 2]; }
};

struct ADLGet {
  long x[8]{};
};

template <__SIZE_TYPE__ I>
constexpr long& get(ADLGet& a) { return a.x[I * 2]; }
} // namespace get_decomposition

template <>
struct tuple_size<get_decomposition::MemberGet> {
  static constexpr __SIZE_TYPE__ value = 3;
};

template <__SIZE_TYPE__ I>
struct tuple_element<I, get_decomposition::MemberGet> {
  using type = int;
};

template <>
struct tuple_size<get_decomposition::ADLGet> {
  static constexpr __SIZE_TYPE__ value = 4;
};

template <__SIZE_TYPE__ I>
struct tuple_element<I, get_decomposition::ADLGet> {
  using type = long;
};

constexpr int member() {
  get_decomposition::MemberGet m;
  int v = 1;
  template for (int& i : m) {
    i = v;
    v++;
  }
  return m.x[0] + m.x[2] + m.x[4];
}

constexpr long adl() {
  get_decomposition::ADLGet m;
  long v = 1;
  template for (long& i : m) {
    i = v;
    v++;
  }
  return m.x[0] + m.x[2] + m.x[4] + m.x[6];
}

static_assert(member() == 6);
static_assert(adl() == 10);

struct TupleSizeOnly {};

template <>
struct tuple_size<TupleSizeOnly> {
  static constexpr __SIZE_TYPE__ value = 3;
};

struct TupleSizeAndGet {
  template <__SIZE_TYPE__>
  constexpr int get() { return 1; }
};

template <>
struct tuple_size<TupleSizeAndGet> {
  static constexpr __SIZE_TYPE__ value = 3;
};

void invalid() {
  template for (auto x : TupleSizeOnly()) {} // expected-error {{use of undeclared identifier 'get'}} \
                                                expected-note {{in implicit initialization of binding declaration}}

  template for (auto x : TupleSizeAndGet()) {} // expected-error {{implicit instantiation of undefined template 'std::tuple_element<0, std::TupleSizeAndGet>'}} \
                                                  expected-note {{in implicit initialization of binding declaration}}
}
} // namespace std

constexpr int generic_lambda() {
  static constexpr int arr[]{1, 2, 3};
  int sum = 0;
  [n = 5, &sum]<class = void>() {
    template for (constexpr auto x : arr) {
      sum += n + x;
    }
  }();
  return sum;
}

static_assert(generic_lambda() == 21);

void for_range_decl_must_be_var() {
  template for (void q() : "error") // expected-error {{expansion statement declaration must declare a variable}}
    ;
}

void init_list_bad() {
  template for (auto y : {{1}, {2}, {3, {4}}, {{{5}}}}); // expected-error {{cannot deduce actual type for variable 'y' with type 'auto' from initializer list}} \
                                                            expected-note {{in instantiation of expansion statement requested here}}
}

// Test that the init statement is evaluated even if the expansion statement
// expands to nothing.
constexpr int init_stmt_empty_expansion() {
#if 0 // Disabled until we support iterating expansion statements.
  static constexpr String empty{""};
#endif // 0
  int x = 0;
  template for (int _ = x += 1; auto i : {}) {}
#if 0 // Disabled until we support iterating expansion statements.
  template for (int _ = x += 2; auto i : empty) {}
#endif // 0
  template for (int _ = x += 3; auto i : Empty()) {}
  return x;
}

static_assert(init_stmt_empty_expansion() == 4);

void vla(int n) {
  int a[n];
  template for (int x : a) {} // expected-error {{cannot expand variable length array type 'int[n]'}}
}

template <typename T>
void template_vla(T& a) { // expected-note {{variably modified type 'int[n]' cannot be used as a template argument}}
  template for (int x : a) {}
}

void instantiate_template_vla(int n) {
  int a[n];
  template_vla(a); // expected-error {{no matching function for call to 'template_vla'}}
}

struct Incomplete; // expected-note 2 {{forward declaration of 'Incomplete'}}
void incomplete_type(Incomplete& s) {
  template for (int x : s) {} // expected-error {{cannot expand expression of incomplete type 'Incomplete'}}
}

template <typename T>
void dependent_incomplete_type(T& s) {
  template for (int x : s) {} // expected-error {{cannot expand expression of incomplete type 'Incomplete'}}
}

template void dependent_incomplete_type<Incomplete>(Incomplete&); // expected-note {{in instantiation of function template specialization 'dependent_incomplete_type<Incomplete>' requested here}}

template <typename T>
void lambda_template(T a) {
  template for (auto x : a) {} // expected-error {{cannot expand lambda closure type}}
}

void lambda_template_call() {
  lambda_template([]{}); // expected-note {{in instantiation of function template specialization}}
}

#if 0 // Disabled until we support iterating expansion statements.
// CWG 3131 makes it possible to expand over non-constexpr ranges.
namespace cwg3131 {
constexpr int f1() {
  int j = 0;
  template for (auto i : Array<int, 3>{1, 2, 3}) j +=i;
  return j;
}

constexpr int f2() {
  Array<int, 3> a{1, 2, 3};
  int j = 0;
  template for (auto i : a) j +=i; // new-interp-error {{expansion statement size is not a constant expression}} \
                                      new-interp-note {{initializer of '__range1' is not a constant expression}} \
                                      new-interp-note {{declared here}}
  return j;
}

static_assert(f1() == 6);
static_assert(f2() == 6); // new-interp-error {{static assertion failed due to requirement 'f2() == 6'}} \
                             new-interp-note {{expression evaluates to '0 == 6'}}

template <typename T>
struct Span {
  T* data;
  __SIZE_TYPE__ size;

  template <__SIZE_TYPE__ N>
  constexpr Span(T(&a)[N]) : data{a}, size{N} {}

  constexpr auto begin() const -> T* { return data; }
  constexpr auto end() const -> T* { return data + size; }
};

constexpr int arr[3] = { 1, 2, 3 };
consteval Span<const int> foo() {
  return Span<const int>(arr);
}

constexpr int f3() {
  int r = 0;
  template for (constexpr auto m : foo())
    r += m;
  return r;
}

static_assert(f3() == 6);
}

// Test that we actually do 'begin + decltype(begin - begin){i}'.
namespace cwg3044 {
struct DifferenceType {
  int v;
  explicit constexpr DifferenceType(__PTRDIFF_TYPE__ v) : v(int(v)) {}
};

struct Range {
  struct iterator {
    DifferenceType v;

    constexpr iterator& operator++() {
      v.v++;
      return *this;
    }

    constexpr int operator*() const { return int(v.v); }
    friend constexpr bool operator!=(iterator a, iterator b) { return a.v.v != b.v.v; }
    friend constexpr iterator operator+(iterator a, DifferenceType b) {
      return iterator{DifferenceType{int(a.v.v) + int(b.v)}};
    }
    friend constexpr DifferenceType operator-(iterator a, iterator b) {
      return DifferenceType{int(a.v.v) - int(b.v.v)};
    }
  };

  constexpr auto begin() const { return iterator{DifferenceType{1}}; }
  constexpr auto end() const { return iterator{DifferenceType{5}}; }
};

constexpr int f() {
  int val = 0;
  template for (auto v : Range()) val += v;
  template for (constexpr auto v : Range()) val += v;
  return val;
}

static_assert(f() == 20);
}

// Test that 'iter' is an lvalue, because we used to not actually create a
// variable for it.
namespace cwg3140 {
struct IterLValue {
  struct iterator {
    int v;

    constexpr iterator& operator++() {
      v++;
      return *this;
    }

    constexpr int operator*() const & { return int(v); }
    constexpr int operator*() const && = delete("'iter' must be an lvalue");
    friend constexpr bool operator!=(iterator a, iterator b) { return a.v != b.v; }
    friend constexpr iterator operator+(iterator a, int b) { return iterator{a.v + b}; }
    friend constexpr int operator-(iterator a, iterator b) { return a.v - b.v; }
  };

  constexpr auto begin() const { return iterator{1}; }
  constexpr auto end() const { return iterator{5}; }
};

constexpr int f() {
  int val = 0;
  template for (auto v : IterLValue()) val += v;
  template for (constexpr auto v : IterLValue()) val += v;
  return val;
}

static_assert(f() == 20);
}
#endif // 0

namespace cwg3149 {
struct NotCopyable {
  int x;
  constexpr NotCopyable(int x) : x{x} {}
  constexpr NotCopyable(NotCopyable&& o) : x{o.x} {}
  NotCopyable(const NotCopyable&) = delete; // expected-note 2 {{explicitly marked deleted here}}
};

struct NotMovable {
  int x;
  constexpr NotMovable(int x) : x{x} {}
  constexpr NotMovable(const NotMovable& o) : x{o.x} {}
  NotMovable(NotMovable&&) = delete; // expected-note 2 {{explicitly marked deleted here}}
};

template <typename T>
struct Wrapper {
  T a;
  T b;
};

constexpr int f() {
  Wrapper<NotMovable> nm{{3},{4}};
  Wrapper<NotCopyable> nc{{5}, {6}};
  int sum = 0;
  template for (auto x : Wrapper<NotCopyable>{{1}, {2}}) sum += x.x;
  template for (auto& x : nc) sum += x.x;
  template for (auto x : static_cast<Wrapper<NotCopyable>&&>(nc)) sum += x.x;
  template for (auto&& x : static_cast<Wrapper<NotMovable>&&>(nm)) sum += x.x;
  return sum;
}

static_assert(f() == 32);

int err() {
  Wrapper<NotCopyable> nc{{3},{4}};
  int sum = 0;

  // expected-error@+2 2 {{call to deleted constructor of 'cwg3149::NotMovable'}}
  // expected-note@+1 2 {{in instantiation of expansion statement requested here}}
  template for (auto x : Wrapper<NotMovable>{{1}, {2}}) sum += x.x;

  // expected-error@+2 2 {{call to deleted constructor of 'cwg3149::NotCopyable'}}
  // expected-note@+1 2 {{in instantiation of expansion statement requested here}}
  template for (auto x : nc) sum += x.x;
  return sum;
}
}

namespace cwg3045 {
void f1() {
  int x;
  int y;

  template for (auto x : {1}) { // expected-note {{previous definition is here}}
    int x{}; // expected-error {{redefinition of 'x'}}
  }

  { int x; }

  template for (auto x : {1}) {
    { int x{}; }
    {
      int x{};
      { int x{}; }
    }
  }

  template for (auto x : {1}) {
    template for (auto y : {1}) { // expected-note {{previous definition is here}}
      int x{};
      int y{}; // expected-error {{redefinition of 'y'}}
    }
  }
}

void f2(int q) {
  switch (q) {
    case 1: template for (auto x : {1}) [[fallthrough]]; // expected-error {{fallthrough annotation does not directly precede switch label}}
  }
  switch (q) {
    case 1: template for (auto x : {1}) [[fallthrough]]; // expected-error {{fallthrough annotation does not directly precede switch label}}
    case 2:;
  }

  switch (q) {
    case 1: template for (auto x : {1}) { [[fallthrough]]; } // expected-error {{fallthrough annotation does not directly precede switch label}}
  }

  switch (q) {
    case 1: template for (auto x : {1}) {
      switch (q) {
        case 1: [[fallthrough]];
        case 2:;
      }
    }
  }
}
}

#if 0 // Disabled until we support iterating expansion statements.
// Check that we apply lifetime extension to iterating expansions statements.
//
// The new constant interpreter erroring on this is likely https://github.com/llvm/llvm-project/issues/187775.
namespace cwg3140 {
constexpr const char* arr = "1";
struct T {
  int& x;
  constexpr T(int& x) noexcept : x(x) {}
  constexpr ~T() noexcept { x = 42; }
  constexpr const char* begin() const { return arr; }
  constexpr const char* end() const { return arr + 1; }
};

constexpr const T& f(const T& t) noexcept { return t; }
constexpr T g(int& x) noexcept { return T(x); }

constexpr int lifetime_extension_iterating() {
  int x = 5;
  int sum  = 0;
  // new-interp-error@+3 {{expansion statement size is not a constant expression}}
  // new-interp-note@+2 {{initializer of '__range1' is not a constant expression}}
  // new-interp-note@+1 {{declared here}}
  template for (auto e : f(g(x))) {
    sum += x;
  }
  return sum + x;
}

template <typename T>
constexpr int lifetime_extension_iterating_dependent() {
  int x = 5;
  int sum  = 0;
  // new-interp-error@+3 {{expansion statement size is not a constant expression}}
  // new-interp-note@+2 {{initializer of '__range0' is not a constant expression}}
  // new-interp-note@+1 {{declared here}}
  template for (auto e : T(f(g(x)))) {
    sum += x;
  }
  return sum + x;
}

template <typename T>
constexpr int lifetime_extension_iterating_instantiation() {
  int x = 5;
  int sum  = 0;
  // new-interp-error@+3 {{expansion statement size is not a constant expression}}
  // new-interp-note@+2 {{initializer of '__range2' is not a constant expression}}
  // new-interp-note@+1 {{declared here}}
  template for (auto e : f(g(x))) {
    sum += x;
  }
  return sum + x;
}

static_assert(lifetime_extension_iterating() == 47); // new-interp-error {{static assertion failed}} new-interp-note {{expression evaluates to}}
static_assert(lifetime_extension_iterating_dependent<const T&>() == 47); // new-interp-error {{static assertion expression is not an integral constant expression}} new-interp-note {{in instantiation of}}
static_assert(lifetime_extension_iterating_instantiation<void>() == 47); // new-interp-error {{static assertion expression is not an integral constant expression}} new-interp-note {{in instantiation of}}
}
#endif // 0

// Tests that make sure that certain parts of an expansion statement end up in
// the right DeclContext.
namespace decl_context {
struct S { int x{}, y{}; };

void dependent_context() {
  // The init-statement should never be in a dependent context.
  template for (int _ = ({ static_assert(false); 4; }); auto x : {}) {} // expected-error {{static assertion failed}}
  template for (({ static_assert(false); }); auto x : {}) {} // expected-error {{static assertion failed}}

  // Likewise, the expansion-initializer is not dependent.
  template for (auto x : { ({ static_assert(false); 4; }) }) {} // expected-error {{static assertion failed}}
  template for (auto x : ({ static_assert(false); S(); })) {} // expected-error {{static assertion failed}}

  // The for-range-declaration *is* dependent because it only appears within the
  // expansion(s), which means that it is discarded entirely if the expansion
  // size is 0.
  template for (decltype(({ static_assert(false); 42; })) _ : {}) {}
  template for (decltype(({ static_assert(false); 42; })) _ : {1}) {} // expected-error {{static assertion failed}} expected-note {{in instantiation of expansion statement}}
}

template <typename>
void not_instantiated() {
  template for (int _ = ({ static_assert(false); 4; }); auto x : {}) {}
  template for (({ static_assert(false); }); auto x : {}) {}
  template for (auto x : { ({ static_assert(false); 4; }) }) {}
  template for (auto x : ({ static_assert(false); S(); })) {}
  template for (decltype(({ static_assert(false); 42; })) _ : {}) {}
  template for (decltype(({ static_assert(false); 42; })) _ : {1}) {}
}

template <typename>
void instantiated() {
  template for (int _ = ({ static_assert(false); 4; }); auto x : {}) {} // expected-error {{static assertion failed}}
  template for (({ static_assert(false); }); auto x : {}) {} // expected-error {{static assertion failed}}
  template for (auto x : { ({ static_assert(false); 4; }) }) {} // expected-error {{static assertion failed}}
  template for (auto x : ({ static_assert(false); S(); })) {} // expected-error {{static assertion failed}}
  template for (decltype(({ static_assert(false); 42; })) _ : {}) {}
  template for (decltype(({ static_assert(false); 42; })) _ : {1}) {} // expected-error {{static assertion failed}}
}

template <typename>
struct Template {
  template <typename>
  void not_fully_instantiated() {
    template for (int _ = ({ static_assert(false); 4; }); auto x : {}) {}
    template for (({ static_assert(false); }); auto x : {}) {}
    template for (auto x : { ({ static_assert(false); 4; }) }) {}
    template for (auto x : ({ static_assert(false); S(); })) {}
    template for (decltype(({ static_assert(false); 42; })) _ : {}) {}
    template for (decltype(({ static_assert(false); 42; })) _ : {1}) {}
  }

  template <typename>
  void instantiated() {
    template for (int _ = ({ static_assert(false); 4; }); auto x : {}) {} // expected-error {{static assertion failed}}
    template for (({ static_assert(false); }); auto x : {}) {} // expected-error {{static assertion failed}}
    template for (auto x : { ({ static_assert(false); 4; }) }) {} // expected-error {{static assertion failed}}
    template for (auto x : ({ static_assert(false); S(); })) {} // expected-error {{static assertion failed}}
    template for (decltype(({ static_assert(false); 42; })) _ : {}) {}
    template for (decltype(({ static_assert(false); 42; })) _ : {1}) {} // expected-error {{static assertion failed}}
  }
};

void f() {
  instantiated<void>(); // expected-note {{in instantiation of function template specialization 'decl_context::instantiated<void>'}}

  // This should *not* produce any diagnostics.
  Template<int>();

  // Rather, the diagnostics are only emitted if we actually fully instantiate
  // the function template.
  Template<void>().instantiated<void>(); // expected-note {{in instantiation of function template specialization 'decl_context::Template<void>::instantiated<void>'}}
}
}

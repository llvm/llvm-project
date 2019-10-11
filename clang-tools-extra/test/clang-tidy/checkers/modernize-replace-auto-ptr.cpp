// RUN: %check_clang_tidy %s modernize-replace-auto-ptr %t -- -- -I %S/Inputs/modernize-replace-auto-ptr

// CHECK-FIXES: #include <utility>

#include "memory.h"

// Instrumentation for auto_ptr_ref test.
struct Base {};
struct Derived : Base {};
std::auto_ptr<Derived> create_derived_ptr();
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: auto_ptr is deprecated, use unique_ptr instead [modernize-replace-auto-ptr]
// CHECK-FIXES: std::unique_ptr<Derived> create_derived_ptr();


// Test function return values (declaration)
std::auto_ptr<char> f_5();
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: auto_ptr is deprecated
// CHECK-FIXES: std::unique_ptr<char> f_5()


// Test function parameters.
void f_6(std::auto_ptr<int>);
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: auto_ptr is deprecated
// CHECK-FIXES: void f_6(std::unique_ptr<int>);
void f_7(const std::auto_ptr<int> &);
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: auto_ptr is deprecated
// CHECK-FIXES: void f_7(const std::unique_ptr<int> &);


// Test on record type fields.
struct A {
  std::auto_ptr<int> field;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: auto_ptr is deprecated
  // CHECK-FIXES: std::unique_ptr<int> field;

  typedef std::auto_ptr<int> int_ptr_type;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: auto_ptr is deprecated
  // CHECK-FIXES: typedef std::unique_ptr<int> int_ptr_type;
};


// FIXME: Test template WITH instantiation.
template <typename T> struct B {
  typedef typename std::auto_ptr<T> created_type;
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: auto_ptr is deprecated
  // CHECK-FIXES: typedef typename std::unique_ptr<T> created_type;

  created_type create() { return std::auto_ptr<T>(new T()); }
  // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: auto_ptr is deprecated
  // CHECK-FIXES: created_type create() { return std::unique_ptr<T>(new T()); }
};


// Test 'using' in a namespace (declaration)
namespace ns_1 {
// Test multiple using declarations.
  using std::auto_ptr;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: auto_ptr is deprecated
  // CHECK-FIXES: using std::unique_ptr;
  using std::auto_ptr;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: auto_ptr is deprecated
  // CHECK-FIXES: using std::unique_ptr;
}


namespace ns_2 {
template <typename T> struct auto_ptr {};
}

void f_1() {
  std::auto_ptr<int> a;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: auto_ptr is deprecated
  // CHECK-FIXES: std::unique_ptr<int> a;

  // Check that spaces aren't modified unnecessarily.
  std:: auto_ptr <int> b;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: auto_ptr is deprecated
  // CHECK-FIXES: std:: unique_ptr <int> b;
  std :: auto_ptr < char > c(new char());
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: auto_ptr is deprecated
  // CHECK-FIXES: std :: unique_ptr < char > c(new char());

  // Test construction from a temporary.
  std::auto_ptr<char> d = std::auto_ptr<char>();
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: auto_ptr is deprecated
  // CHECK-MESSAGES: :[[@LINE-2]]:32: warning: auto_ptr is deprecated
  // CHECK-FIXES: std::unique_ptr<char> d = std::unique_ptr<char>();

  typedef std::auto_ptr<int> int_ptr_t;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: auto_ptr is deprecated
  // CHECK-FIXES: typedef std::unique_ptr<int> int_ptr_t;
  int_ptr_t e(new int());

  // Test pointers.
  std::auto_ptr<int> *f;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: auto_ptr is deprecated
  // CHECK-FIXES: std::unique_ptr<int> *f;

  // Test 'static' declarations.
  static std::auto_ptr<int> g;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: auto_ptr is deprecated
  // CHECK-FIXES: static std::unique_ptr<int> g;

  // Test with cv-qualifiers.
  const std::auto_ptr<int> h;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: auto_ptr is deprecated
  // CHECK-FIXES: const std::unique_ptr<int> h;
  volatile std::auto_ptr<int> i;
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: auto_ptr is deprecated
  // CHECK-FIXES: volatile std::unique_ptr<int> i;
  const volatile std::auto_ptr<int> j;
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: auto_ptr is deprecated
  // CHECK-FIXES: const volatile std::unique_ptr<int> j;

  // Test auto and initializer-list.
  auto k = std::auto_ptr<int>{};
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: auto_ptr is deprecated
  // CHECK-FIXES: auto k = std::unique_ptr<int>{};
  std::auto_ptr<int> l{std::auto_ptr<int>()};
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: auto_ptr is deprecated
  // CHECK-MESSAGES: :[[@LINE-2]]:29: warning: auto_ptr is deprecated
  // CHECK-FIXES: std::unique_ptr<int> l{std::unique_ptr<int>()};

  // Test interlocked auto_ptr.
  std::auto_ptr<std::auto_ptr<int> > m;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: auto_ptr is deprecated
  // CHECK-MESSAGES: :[[@LINE-2]]:22: warning: auto_ptr is deprecated
  // CHECK-FIXES: std::unique_ptr<std::unique_ptr<int> > m;

  // Test temporaries.
  std::auto_ptr<char>();
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: auto_ptr is deprecated
  // CHECK-FIXES: std::unique_ptr<char>();

  // Test void-specialization.
  std::auto_ptr<void> n;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: auto_ptr is deprecated
  // CHECK-FIXES: std::unique_ptr<void> n;

  // Test template WITH instantiation (instantiation).
  B<double> o;
  std::auto_ptr<double> p(o.create());
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: auto_ptr is deprecated
  // CHECK-FIXES: std::unique_ptr<double> p(o.create());

  // Test 'using' in a namespace ("definition").
  ns_1::auto_ptr<int> q;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: auto_ptr is deprecated
  // CHECK-FIXES: ns_1::unique_ptr<int> q;

  // Test construction with an 'auto_ptr_ref'.
  std::auto_ptr<Base> r(create_derived_ptr());
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: auto_ptr is deprecated
  // CHECK-FIXES: std::unique_ptr<Base> r(create_derived_ptr());
}

// Test without the nested name specifiers.
void f_2() {
  using namespace std;

  auto_ptr<int> a;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: auto_ptr is deprecated
  // CHECK-FIXES: unique_ptr<int> a;
}

// Test using declaration.
void f_3() {
  using std::auto_ptr;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: auto_ptr is deprecated
  // CHECK-FIXES: using std::unique_ptr;

  auto_ptr<int> a;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: auto_ptr is deprecated
  // CHECK-FIXES: unique_ptr<int> a;
}

// Test messing-up with macros.
void f_4() {
#define MACRO_1 <char>
  std::auto_ptr MACRO_1 p(new char());
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: auto_ptr is deprecated
  // CHECK-FIXES: std::unique_ptr MACRO_1 p(new char());
#define MACRO_2 auto_ptr
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: auto_ptr is deprecated
  // CHECK-FIXES: #define MACRO_2 unique_ptr
  std::MACRO_2<int> q;
#define MACRO_3(Type) std::auto_ptr<Type>
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: auto_ptr is deprecated
  // CHECK-FIXES: #define MACRO_3(Type) std::unique_ptr<Type>
  MACRO_3(float)r(new float());
#define MACRO_4 std::auto_ptr
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: auto_ptr is deprecated
  // CHECK-FIXES: #define MACRO_4 std::unique_ptr
  using MACRO_4;
#undef MACRO_1
#undef MACRO_2
#undef MACRO_3
#undef MACRO_4
}

// Test function return values (definition).
std::auto_ptr<char> f_5()
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: auto_ptr is deprecated
  // CHECK-FIXES: std::unique_ptr<char> f_5()
{
  // Test constructor.
  return std::auto_ptr<char>(new char());
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: auto_ptr is deprecated
  // CHECK-FIXES: return std::unique_ptr<char>(new char());
}

// Test that non-std auto_ptr aren't replaced.
void f_8() {
  ns_2::auto_ptr<char> a;
  using namespace ns_2;
  auto_ptr<int> b;
}

// Fail to modify when the template is never instantiated.
//
// This might not be an issue. If it's never used it doesn't really matter if
// it's changed or not. If it's a header and one of the source use it, then it
// will still be changed.
template <typename X>
void f() {
  std::auto_ptr<X> p;
}

// FIXME: Alias template could be replaced if a matcher existed.
namespace std {
template <typename T> using aaaaaaaa = auto_ptr<T>;
}

// We want to avoid replacing 'aaaaaaaa' by unique_ptr here. It's better to
// change the type alias directly.
std::aaaaaaaa<int> d;


void takes_ownership_fn(std::auto_ptr<int> x);
// CHECK-MESSAGES: :[[@LINE-1]]:30: warning: auto_ptr is deprecated
// CHECK-FIXES: void takes_ownership_fn(std::unique_ptr<int> x);

std::auto_ptr<int> get_by_value();
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: auto_ptr is deprecated
// CHECK-FIXES: std::unique_ptr<int> get_by_value();

class Wrapper {
 public:
  std::auto_ptr<int> &get_wrapped();
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: auto_ptr is deprecated

 private:
  std::auto_ptr<int> wrapped;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: auto_ptr is deprecated
};

void f() {
  std::auto_ptr<int> a, b, c;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: auto_ptr is deprecated
  // CHECK-FIXES: std::unique_ptr<int> a, b, c;
  Wrapper wrapper_a, wrapper_b;

  a = b;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use std::move to transfer ownership
  // CHECK-FIXES: a = std::move(b);

  wrapper_a.get_wrapped() = wrapper_b.get_wrapped();
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: use std::move to transfer ownership
  // CHECK-FIXES: wrapper_a.get_wrapped() = std::move(wrapper_b.get_wrapped());

  // Test that 'std::move()' is inserted when call to the
  // copy-constructor are made.
  takes_ownership_fn(c);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: use std::move to transfer ownership
  // CHECK-FIXES: takes_ownership_fn(std::move(c));
  takes_ownership_fn(wrapper_a.get_wrapped());
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: use std::move to transfer ownership
  // CHECK-FIXES: takes_ownership_fn(std::move(wrapper_a.get_wrapped()));

  std::auto_ptr<int> d[] = { std::auto_ptr<int>(new int(1)),
                             std::auto_ptr<int>(new int(2)) };
  // CHECK-MESSAGES: :[[@LINE-2]]:8: warning: auto_ptr is deprecated
  // CHECK-MESSAGES: :[[@LINE-3]]:35: warning: auto_ptr is deprecated
  // CHECK-MESSAGES: :[[@LINE-3]]:35: warning: auto_ptr is deprecated
  // CHECK-FIXES: std::unique_ptr<int> d[] = { std::unique_ptr<int>(new int(1)),
  // CHECK-FIXES-NEXT:                         std::unique_ptr<int>(new int(2)) };
  std::auto_ptr<int> e = d[0];
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: auto_ptr is deprecated
  // CHECK-MESSAGES: :[[@LINE-2]]:26: warning: use std::move to transfer ownership
  // CHECK: std::unique_ptr<int> e = std::move(d[0]);

  // Test that std::move() is not used when assigning an rvalue
  std::auto_ptr<int> f;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: auto_ptr is deprecated
  // CHECK-FIXES: std::unique_ptr<int> f;
  f = std::auto_ptr<int>(new int(0));
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: auto_ptr is deprecated
  // CHECK-NEXT: f = std::unique_ptr<int>(new int(0));

  std::auto_ptr<int> g = get_by_value();
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: auto_ptr is deprecated
  // CHECK-FIXES: std::unique_ptr<int> g = get_by_value();
}

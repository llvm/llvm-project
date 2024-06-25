// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fsyntax-only -verify -std=c++11 %s -fexperimental-new-constant-interpreter

static int test0 __attribute__((weak)); // expected-error {{weak declaration cannot have internal linkage}}
static void test1() __attribute__((weak)); // expected-error {{weak declaration cannot have internal linkage}}

namespace test2 __attribute__((weak)) { // expected-warning {{'weak' attribute only applies to variables, functions, and classes}}
}

namespace {
  int test3 __attribute__((weak)); // expected-error {{weak declaration cannot have internal linkage}}
  void test4() __attribute__((weak)); // expected-error {{weak declaration cannot have internal linkage}}
}

struct Test5 {
  static void test5() __attribute__((weak)); // no error
};

namespace {
  struct Test6 {
    static void test6() __attribute__((weak)); // expected-error {{weak declaration cannot have internal linkage}}
  };
}

// GCC rejects the instantiation with the internal type, but some existing
// code expects it. It is also not that different from giving hidden visibility
// to parts of a template that have explicit default visibility, so we accept
// this.
template <class T> struct Test7 {
  void test7() __attribute__((weak)) {}
  static int var __attribute__((weak));
};
template <class T>
int Test7<T>::var;
namespace { class Internal {}; }
template struct Test7<Internal>;
template struct Test7<int>;

class __attribute__((weak)) Test8 {}; // OK

__attribute__((weak)) auto Test9 = Internal(); // expected-error {{weak declaration cannot have internal linkage}}

[[gnu::weak]] void weak_function();
struct WithWeakMember {
  [[gnu::weak]] void weak_method();
  [[gnu::weak]] virtual void virtual_weak_method();
};
constexpr bool weak_function_is_non_null = &weak_function != nullptr; // expected-error {{must be initialized by a constant expression}}
// expected-note@-1 {{comparison against address of weak declaration '&weak_function' can only be performed at runtime}}
constexpr bool weak_method_is_non_null = &WithWeakMember::weak_method != nullptr; // expected-error {{must be initialized by a constant expression}}
// expected-note@-1 {{comparison against pointer to weak member 'WithWeakMember::weak_method' can only be performed at runtime}}
// GCC accepts this and says the result is always non-null. That's consistent
// with the ABI rules for member pointers, but seems unprincipled, so we do not
// follow it for now.
// TODO: Consider warning on such comparisons, as they do not test whether the
// virtual member function is present.
constexpr bool virtual_weak_method_is_non_null = &WithWeakMember::virtual_weak_method != nullptr; // expected-error {{must be initialized by a constant expression}}
// expected-note@-1 {{comparison against pointer to weak member 'WithWeakMember::virtual_weak_method' can only be performed at runtime}}

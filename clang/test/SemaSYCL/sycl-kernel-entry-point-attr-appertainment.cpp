// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fsyntax-only -fsycl-is-device -verify %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++20 -fsyntax-only -fsycl-is-device -verify %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++23 -fsyntax-only -fsycl-is-device -verify %s

// These tests validate appertainment for the sycl_kernel_entry_point attribute.

#if __cplusplus >= 202002L
// Mock coroutine support.
namespace std {

template<typename Promise = void>
struct coroutine_handle {
  template<typename T>
  coroutine_handle(const coroutine_handle<T>&);
  static coroutine_handle from_address(void *addr);
};

template<typename R, typename... Args>
struct coroutine_traits {
  struct suspend_never {
    bool await_ready() const noexcept;
    void await_suspend(std::coroutine_handle<>) const noexcept;
    void await_resume() const noexcept;
  };
  struct promise_type {
    void get_return_object() noexcept;
    suspend_never initial_suspend() const noexcept;
    suspend_never final_suspend() const noexcept;
    void return_void() noexcept;
    void unhandled_exception() noexcept;
  };
};

}
#endif

// A unique kernel name type is required for each declared kernel entry point.
template<int, int = 0> struct KN;


////////////////////////////////////////////////////////////////////////////////
// Valid declarations.
////////////////////////////////////////////////////////////////////////////////

// Function declaration with GNU attribute spelling
// expected-warning@+1 {{unknown attribute 'sycl_kernel_entry_point' ignored}}
__attribute__((sycl_kernel_entry_point(KN<1>)))
void ok1();

// Function declaration with C++11 attribute spelling.
[[clang::sycl_kernel_entry_point(KN<2>)]]
void ok2();

// Function definition.
[[clang::sycl_kernel_entry_point(KN<3>)]]
void ok3() {}

// Function template definition.
template<typename KNT, typename T>
[[clang::sycl_kernel_entry_point(KNT)]]
void ok4(T) {}

// Function template explicit specialization.
template<>
[[clang::sycl_kernel_entry_point(KN<4,1>)]]
void ok4<KN<4,1>>(int) {}

// Function template explicit instantiation.
template void ok4<KN<4,2>, long>(long);

namespace NS {
// Function declaration at namespace scope.
[[clang::sycl_kernel_entry_point(KN<5>)]]
void ok5();
}

struct S6 {
  // Static member function declaration.
  [[clang::sycl_kernel_entry_point(KN<6>)]]
  static void ok6();
};

// Dependent hidden friend definition.
template<typename KNT>
struct S7 {
  [[clang::sycl_kernel_entry_point(KNT)]]
  friend void ok7(S7) {}
};
void test_ok7() {
  ok7(S7<KN<7>>{});
}

// Non-dependent hidden friend definition.
struct S8Base {};
template<typename>
struct S8 : S8Base {
  [[clang::sycl_kernel_entry_point(KN<8>)]]
  friend void ok8(const S8Base&) {}
};
void test_ok8() {
  ok8(S8<int>{});
}

// The sycl_kernel_entry_point attribute must match across declarations and
// cannot be added for the first time after a definition.
[[clang::sycl_kernel_entry_point(KN<9>)]]
void ok9();
[[clang::sycl_kernel_entry_point(KN<9>)]]
void ok9();
[[clang::sycl_kernel_entry_point(KN<10>)]]
void ok10();
void ok10() {}
void ok11();
[[clang::sycl_kernel_entry_point(KN<11>)]]
void ok11() {}

using VOID = void;
[[clang::sycl_kernel_entry_point(KN<12>)]]
VOID ok12();
[[clang::sycl_kernel_entry_point(KN<13>)]]
const void ok13();

#if __cplusplus >= 202302L
auto ok14 = [] [[clang::sycl_kernel_entry_point(KN<14>)]] static -> void {};
#endif

template<typename KNT, typename T>
struct S15 {
  // Don't diagnose a dependent return type as a non-void type.
  [[clang::sycl_kernel_entry_point(KNT)]]
  static T ok15();
};


////////////////////////////////////////////////////////////////////////////////
// Invalid declarations.
////////////////////////////////////////////////////////////////////////////////

// The sycl_kernel_entry_point attribute cannot appertain to main() because
// main() has a non-void return type. However, if the requirement for a void
// return type were to be relaxed or if an allowance was made for main() to
// return void (as gcc allows in some modes and as has been proposed to WG21
// on occassion), main() still can't function as a SYCL kernel entry point,
// so this test ensures such attempted uses of the attribute are rejected.
struct Smain;
// expected-error@+1 {{'clang::sycl_kernel_entry_point' attribute only applies to functions with a 'void' return type}}
[[clang::sycl_kernel_entry_point(Smain)]]
int main();

template<int> struct BADKN;

struct B1 {
  // Non-static data member declaration.
  // expected-error@+1 {{'clang::sycl_kernel_entry_point' attribute only applies to functions}}
  [[clang::sycl_kernel_entry_point(BADKN<1>)]]
  int bad1;
};

struct B2 {
  // Static data member declaration.
  // expected-error@+1 {{'clang::sycl_kernel_entry_point' attribute only applies to functions}}
  [[clang::sycl_kernel_entry_point(BADKN<2>)]]
  static int bad2;
};

struct B3 {
  // Non-static member function declaration.
  // expected-error@+1 {{the 'clang::sycl_kernel_entry_point' attribute cannot be applied to a non-static member function}}
  [[clang::sycl_kernel_entry_point(BADKN<3>)]]
  void bad3();
};

// expected-error@+1 {{'clang::sycl_kernel_entry_point' attribute only applies to functions}}
namespace [[clang::sycl_kernel_entry_point(BADKN<4>)]] bad4 {}

#if __cplusplus >= 202002L
// expected-error@+2 {{'clang::sycl_kernel_entry_point' attribute only applies to functions}}
template<typename>
concept bad5 [[clang::sycl_kernel_entry_point(BADKN<5>)]] = true;
#endif

// Type alias declarations.
// expected-error@+1 {{'clang::sycl_kernel_entry_point' attribute only applies to functions}}
typedef void bad6 [[clang::sycl_kernel_entry_point(BADKN<6>)]] ();
// expected-error@+1 {{'clang::sycl_kernel_entry_point' attribute only applies to functions}}
using bad7 [[clang::sycl_kernel_entry_point(BADKN<7>)]] = void();
// expected-error@+1 {{'clang::sycl_kernel_entry_point' attribute only applies to functions}}
using bad8 [[clang::sycl_kernel_entry_point(BADKN<8>)]] = int;
// expected-error@+1 {{'clang::sycl_kernel_entry_point' attribute cannot be applied to types}}
using bad9 = int [[clang::sycl_kernel_entry_point(BADKN<9>)]];
// expected-error@+1 {{'clang::sycl_kernel_entry_point' attribute cannot be applied to types}}
using bad10 = int() [[clang::sycl_kernel_entry_point(BADKN<10>)]];

// Variable declaration.
// expected-error@+1 {{'clang::sycl_kernel_entry_point' attribute only applies to functions}}
[[clang::sycl_kernel_entry_point(BADKN<11>)]]
int bad11;

// Class declaration.
// expected-error@+1 {{'clang::sycl_kernel_entry_point' attribute only applies to functions}}
struct [[clang::sycl_kernel_entry_point(BADKN<12>)]] bad12;

// Enumeration declaration.
// expected-error@+1 {{'clang::sycl_kernel_entry_point' attribute only applies to functions}}
enum [[clang::sycl_kernel_entry_point(BADKN<13>)]] bad13 {};

// Enumerator.
// expected-error@+2 {{'clang::sycl_kernel_entry_point' attribute only applies to functions}}
enum {
  bad14 [[clang::sycl_kernel_entry_point(BADKN<14>)]]
};

// Attribute added after the definition.
// expected-error@+3 {{the 'clang::sycl_kernel_entry_point' attribute cannot be added to a function after the function is defined}}
// expected-note@+1 {{previous definition is here}}
void bad15() {}
[[clang::sycl_kernel_entry_point(BADKN<15>)]]
void bad15();

// The function must return void.
// expected-error@+1 {{the 'clang::sycl_kernel_entry_point' attribute only applies to functions with a 'void' return type}}
[[clang::sycl_kernel_entry_point(BADKN<16>)]]
int bad16();

// Function parameters.
// expected-error@+1 {{'clang::sycl_kernel_entry_point' attribute only applies to functions}}
void bad17(void (fp [[clang::sycl_kernel_entry_point(BADKN<17>)]])());

// Function template parameters.
// FIXME: Clang currently ignores attributes that appear in template parameters
// FIXME: and the C++ standard is unclear regarding whether such attributes are
// FIXME: permitted. P3324 (Attributes for namespace aliases, template
// FIXME: parameters, and lambda captures) seeks to clarify the situation.
// FIXME-expected-error@+1 {{'clang::sycl_kernel_entry_point' attribute only applies to functions}}
template<void (fp [[clang::sycl_kernel_entry_point(BADKN<18>)]])()>
void bad18();

#if __cplusplus >= 202002L
// expected-error@+1 {{the 'clang::sycl_kernel_entry_point' attribute cannot be applied to a coroutine}}
[[clang::sycl_kernel_entry_point(BADKN<19>)]]
void bad19() {
  co_return;
}
#endif

struct B20 {
  // expected-error@+1 {{the 'clang::sycl_kernel_entry_point' attribute cannot be applied to a non-static member function}}
  [[clang::sycl_kernel_entry_point(BADKN<20>)]]
  B20();
};

struct B21 {
  // expected-error@+1 {{the 'clang::sycl_kernel_entry_point' attribute cannot be applied to a non-static member function}}
  [[clang::sycl_kernel_entry_point(BADKN<21>)]]
  ~B21();
};

// expected-error@+1 {{the 'clang::sycl_kernel_entry_point' attribute cannot be applied to a variadic function}}
[[clang::sycl_kernel_entry_point(BADKN<22>)]]
void bad22(...);

// expected-error@+1 {{the 'clang::sycl_kernel_entry_point' attribute cannot be applied to a deleted function}}
[[clang::sycl_kernel_entry_point(BADKN<23>)]]
void bad23() = delete;

// expected-error@+1 {{the 'clang::sycl_kernel_entry_point' attribute cannot be applied to a constexpr function}}
[[clang::sycl_kernel_entry_point(BADKN<24>)]]
constexpr void bad24() {}

#if __cplusplus >= 202002L
// expected-error@+1 {{the 'clang::sycl_kernel_entry_point' attribute cannot be applied to a consteval function}}
[[clang::sycl_kernel_entry_point(BADKN<25>)]]
consteval void bad25() {}
#endif

// expected-error@+1 {{the 'clang::sycl_kernel_entry_point' attribute cannot be applied to a function declared with the 'noreturn' attribute}}
[[clang::sycl_kernel_entry_point(BADKN<26>)]]
[[noreturn]] void bad26();

// expected-error@+3 {{attribute 'target' multiversioning cannot be combined with attribute 'clang::sycl_kernel_entry_point'}}
__attribute__((target("avx"))) void bad27();
[[clang::sycl_kernel_entry_point(BADKN<27>)]]
__attribute__((target("sse4.2"))) void bad27();

template<typename KNT>
struct B28 {
  // expected-error@+1 {{the 'clang::sycl_kernel_entry_point' attribute cannot be applied to a deleted function}}
  [[clang::sycl_kernel_entry_point(KNT)]]
  friend void bad28() = delete;
};

#if __cplusplus >= 202002L
template<typename KNT, typename T>
struct B29 {
  // expected-error@+1 {{the 'clang::sycl_kernel_entry_point' attribute cannot be applied to a defaulted function}}
  [[clang::sycl_kernel_entry_point(KNT)]]
  friend T operator==(B29, B29) = default;
};
#endif

#if __cplusplus >= 202002L
template<typename KNT>
struct B30 {
  // expected-error@+1 {{the 'clang::sycl_kernel_entry_point' attribute cannot be applied to a coroutine}}
  [[clang::sycl_kernel_entry_point(KNT)]]
  friend void bad30() { co_return; }
};
#endif

template<typename KNT>
struct B31 {
  // expected-error@+1 {{the 'clang::sycl_kernel_entry_point' attribute cannot be applied to a variadic function}}
  [[clang::sycl_kernel_entry_point(KNT)]]
  friend void bad31(...) {}
};

template<typename KNT>
struct B32 {
  // expected-error@+1 {{the 'clang::sycl_kernel_entry_point' attribute cannot be applied to a constexpr function}}
  [[clang::sycl_kernel_entry_point(KNT)]]
  friend constexpr void bad32() {}
};

#if __cplusplus >= 202002L
template<typename KNT>
struct B33 {
  // expected-error@+1 {{the 'clang::sycl_kernel_entry_point' attribute cannot be applied to a consteval function}}
  [[clang::sycl_kernel_entry_point(KNT)]]
  friend consteval void bad33() {}
};
#endif

template<typename KNT>
struct B34 {
  // expected-error@+1 {{the 'clang::sycl_kernel_entry_point' attribute cannot be applied to a function declared with the 'noreturn' attribute}}
  [[clang::sycl_kernel_entry_point(KNT)]]
  [[noreturn]] friend void bad34() {}
};

#if __cplusplus >= 202302L
// expected-error@+1 {{the 'clang::sycl_kernel_entry_point' attribute cannot be applied to a non-static member function}}
auto bad35 = [] [[clang::sycl_kernel_entry_point(BADKN<35>)]] -> void {};
#endif

#if __cplusplus >= 202302L
// expected-error@+1 {{the 'clang::sycl_kernel_entry_point' attribute only applies to functions with a non-deduced 'void' return type}}
auto bad36 = [] [[clang::sycl_kernel_entry_point(BADKN<36>)]] static {};
#endif

#if __cplusplus >= 202302L
// expected-error@+1 {{the 'clang::sycl_kernel_entry_point' attribute cannot be applied to a coroutine}}
auto bad37 = [] [[clang::sycl_kernel_entry_point(BADKN<37>)]] static -> void { co_return; };
#endif

// expected-error@+1 {{the 'clang::sycl_kernel_entry_point' attribute cannot be applied to a function defined with a function try block}}
[[clang::sycl_kernel_entry_point(BADKN<38>)]]
void bad38() try {} catch(...) {}

// expected-error@+2 {{the 'clang::sycl_kernel_entry_point' attribute cannot be applied to a function defined with a function try block}}
template<typename>
[[clang::sycl_kernel_entry_point(BADKN<39>)]]
void bad39() try {} catch(...) {}

// expected-error@+1 {{'clang::sycl_kernel_entry_point' attribute only applies to functions}}
[[clang::sycl_kernel_entry_point(BADKN<40>)]];

void bad41() {
  // expected-error@+1 {{'clang::sycl_kernel_entry_point' attribute cannot be applied to a statement}}
  [[clang::sycl_kernel_entry_point(BADKN<41>)]];
}

struct B42 {
  // expected-warning@+1 {{declaration does not declare anything}}
  [[clang::sycl_kernel_entry_point(BADKN<42>)]];
};

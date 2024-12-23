// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fsyntax-only -fsycl-is-device -verify %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++20 -fsyntax-only -fsycl-is-device -verify %s

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
__attribute__((sycl_kernel_entry_point(KN<1>)))
void ok1();

// Function declaration with Clang attribute spelling.
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

// Dependent friend function.
template<typename KNT>
struct S7 {
  [[clang::sycl_kernel_entry_point(KNT)]]
  friend void ok7(S7) {}
};
void test_ok7() {
  ok7(S7<KN<7>>{});
}

// The sycl_kernel_entry_point attribute must match across declarations and
// cannot be added for the first time after a definition.
[[clang::sycl_kernel_entry_point(KN<8>)]]
void ok8();
[[clang::sycl_kernel_entry_point(KN<8>)]]
void ok8();
[[clang::sycl_kernel_entry_point(KN<9>)]]
void ok9();
void ok9() {}
void ok10();
[[clang::sycl_kernel_entry_point(KN<10>)]]
void ok10() {}

using VOID = void;
[[clang::sycl_kernel_entry_point(KN<11>)]]
VOID ok11();
[[clang::sycl_kernel_entry_point(KN<12>)]]
const void ok12();


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
// expected-error@+1 {{'sycl_kernel_entry_point' attribute only applies to functions with a 'void' return type}}
[[clang::sycl_kernel_entry_point(Smain)]]
int main();

template<int> struct BADKN;

struct B1 {
  // Non-static data member declaration.
  // expected-error@+1 {{'sycl_kernel_entry_point' attribute only applies to functions}}
  [[clang::sycl_kernel_entry_point(BADKN<1>)]]
  int bad1;
};

struct B2 {
  // Static data member declaration.
  // expected-error@+1 {{'sycl_kernel_entry_point' attribute only applies to functions}}
  [[clang::sycl_kernel_entry_point(BADKN<2>)]]
  static int bad2;
};

struct B3 {
  // Non-static member function declaration.
  // expected-error@+1 {{'sycl_kernel_entry_point' attribute cannot be applied to a non-static member function}}
  [[clang::sycl_kernel_entry_point(BADKN<3>)]]
  void bad3();
};

// expected-error@+1 {{'sycl_kernel_entry_point' attribute only applies to functions}}
namespace bad4 [[clang::sycl_kernel_entry_point(BADKN<4>)]] {}

#if __cplusplus >= 202002L
// expected-error@+2 {{'sycl_kernel_entry_point' attribute only applies to functions}}
template<typename>
concept bad5 [[clang::sycl_kernel_entry_point(BADKN<5>)]] = true;
#endif

// Type alias declarations.
// expected-error@+1 {{'sycl_kernel_entry_point' attribute only applies to functions}}
typedef void bad6 [[clang::sycl_kernel_entry_point(BADKN<6>)]] ();
// expected-error@+1 {{'sycl_kernel_entry_point' attribute only applies to functions}}
using bad7 [[clang::sycl_kernel_entry_point(BADKN<7>)]] = void();
// expected-error@+1 {{'sycl_kernel_entry_point' attribute only applies to functions}}
using bad8 [[clang::sycl_kernel_entry_point(BADKN<8>)]] = int;
// expected-error@+1 {{'sycl_kernel_entry_point' attribute cannot be applied to types}}
using bad9 = int [[clang::sycl_kernel_entry_point(BADKN<9>)]];
// expected-error@+1 {{'sycl_kernel_entry_point' attribute cannot be applied to types}}
using bad10 = int() [[clang::sycl_kernel_entry_point(BADKN<10>)]];

// Variable declaration.
// expected-error@+1 {{'sycl_kernel_entry_point' attribute only applies to functions}}
[[clang::sycl_kernel_entry_point(BADKN<11>)]]
int bad11;

// Class declaration.
// expected-error@+1 {{'sycl_kernel_entry_point' attribute only applies to functions}}
struct [[clang::sycl_kernel_entry_point(BADKN<12>)]] bad12;

// Enumeration declaration.
// expected-error@+1 {{'sycl_kernel_entry_point' attribute only applies to functions}}
enum [[clang::sycl_kernel_entry_point(BADKN<13>)]] bad13 {};

// Enumerator.
// expected-error@+2 {{'sycl_kernel_entry_point' attribute only applies to functions}}
enum {
  bad14 [[clang::sycl_kernel_entry_point(BADKN<14>)]]
};

// Attribute added after the definition.
// expected-error@+3 {{'sycl_kernel_entry_point' attribute cannot be added to a function after the function is defined}}
// expected-note@+1 {{previous definition is here}}
void bad15() {}
[[clang::sycl_kernel_entry_point(BADKN<15>)]]
void bad15();

// The function must return void.
// expected-error@+1 {{'sycl_kernel_entry_point' attribute only applies to functions with a 'void' return type}}
[[clang::sycl_kernel_entry_point(BADKN<16>)]]
int bad16();

// Function parameters.
// expected-error@+1 {{'sycl_kernel_entry_point' attribute only applies to functions}}
void bad17(void (fp [[clang::sycl_kernel_entry_point(BADKN<17>)]])());

// Function template parameters.
// expected-error@+1 {{'sycl_kernel_entry_point' attribute only applies to functions}}
template<void (fp [[clang::sycl_kernel_entry_point(BADKN<18>)]])()>
void bad18();

#if __cplusplus >= 202002L
// expected-error@+1 {{'sycl_kernel_entry_point' attribute cannot be applied to a coroutine}}
[[clang::sycl_kernel_entry_point(BADKN<19>)]]
void bad19() {
  co_return;
}
#endif

struct B20 {
  // expected-error@+1 {{'sycl_kernel_entry_point' attribute cannot be applied to a non-static member function}}
  [[clang::sycl_kernel_entry_point(BADKN<20>)]]
  B20();
};

struct B21 {
  // expected-error@+1 {{'sycl_kernel_entry_point' attribute cannot be applied to a non-static member function}}
  [[clang::sycl_kernel_entry_point(BADKN<21>)]]
  ~B21();
};

// expected-error@+1 {{'sycl_kernel_entry_point' attribute cannot be applied to a variadic function}}
[[clang::sycl_kernel_entry_point(BADKN<22>)]]
void bad22(...);

// expected-error@+1 {{'sycl_kernel_entry_point' attribute cannot be applied to a deleted function}}
[[clang::sycl_kernel_entry_point(BADKN<23>)]]
void bad23() = delete;

// expected-error@+1 {{'sycl_kernel_entry_point' attribute cannot be applied to a constexpr function}}
[[clang::sycl_kernel_entry_point(BADKN<24>)]]
constexpr void bad24() {}

#if __cplusplus >= 202002L
// expected-error@+1 {{'sycl_kernel_entry_point' attribute cannot be applied to a consteval function}}
[[clang::sycl_kernel_entry_point(BADKN<25>)]]
consteval void bad25() {}
#endif

// expected-error@+1 {{'sycl_kernel_entry_point' attribute cannot be applied to a function declared with the 'noreturn' attribute}}
[[clang::sycl_kernel_entry_point(BADKN<26>)]]
[[noreturn]] void bad26();

// expected-error@+3 {{attribute 'target' multiversioning cannot be combined with attribute 'sycl_kernel_entry_point'}}
__attribute__((target("avx"))) void bad27();
[[clang::sycl_kernel_entry_point(BADKN<27>)]]
__attribute__((target("sse4.2"))) void bad27();

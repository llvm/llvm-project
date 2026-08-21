// RUN: %clang_cc1 %s -fcuda-is-device -std=c++17 -triple spirv32 -aux-triple x86_64-pc-windows-msvc -fms-extensions -verify
// RUN: %clang_cc1 %s -fcuda-is-device -std=c++17 -triple spirv64 -aux-triple x86_64-pc-windows-msvc -fms-extensions -verify
// RUN: %clang_cc1 %s -fcuda-is-device -std=c++17 -triple spirv64-amd-amdhsa -aux-triple x86_64-pc-windows-msvc -fms-extensions -verify

// expected-no-diagnostics

// __cdecl must be the device default calling convention so std::function<void()>
// matches the library specialization keyed on __cdecl on amdgcnspirv Windows.

template <class T> struct classify_fn {
  static constexpr int value = 0;
};
template <class R, class... A> struct classify_fn<R __cdecl(A...)> {
  static constexpr int value = 1;
};
template <class R, class... A> struct classify_fn<R __vectorcall(A...)> {
  static constexpr int value = 2;
};

// Unadorned function types resolve to the __cdecl specialization.
static_assert(classify_fn<void()>::value == classify_fn<void __cdecl()>::value,
              "The default and __cdecl calling conventions must match");
static_assert(classify_fn<void()>::value == 1, "void() must match __cdecl");

// __vectorcall must not be folded into the device default alongside __cdecl.
static_assert(classify_fn<void __vectorcall()>::value == 2,
              "__vectorcall must match its own specialization, not __cdecl");

// Variadic __cdecl is accepted, as it is on the host, so that prevalent Windows
// headers remain includable during device compilation.
using VariadicCdecl = int __cdecl(int, ...);

struct ThiscallBase {
  virtual void __attribute__((thiscall)) boo();
};
struct ThiscallDerived : ThiscallBase {
  void boo() override;
};

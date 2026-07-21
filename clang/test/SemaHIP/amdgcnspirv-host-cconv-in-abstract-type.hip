// RUN: %clang_cc1 %s -fcuda-is-device -std=c++17 -triple spirv32 -aux-triple x86_64-pc-windows-msvc -fms-extensions -verify
// RUN: %clang_cc1 %s -fcuda-is-device -std=c++17 -triple spirv64 -aux-triple x86_64-pc-windows-msvc -fms-extensions -verify
// RUN: %clang_cc1 %s -fcuda-is-device -std=c++17 -triple spirv64-amd-amdhsa -aux-triple x86_64-pc-windows-msvc -fms-extensions -verify

// __cdecl on an abstract function type must normalize to the device default
// so std::function<void()> matches the library specialization keyed on
// __cdecl on amdgcnspirv Windows.

// expected-no-diagnostics

template <class T> struct classify_fn {
  static constexpr int value = 0;
};
template <class R, class... A> struct classify_fn<R __cdecl(A...)> {
  static constexpr int value = 1;
};
template <class R, class... A> struct classify_fn<R __vectorcall(A...)> {
  static constexpr int value = 2;
};
template <class R> struct classify_fn<R __cdecl(int, ...)> {
  static constexpr int value = 3;
};

// Unadorned function types resolve to the __cdecl specialization.
static_assert(classify_fn<void()>::value == 1, "void() must match __cdecl");
static_assert(classify_fn<int(double, char)>::value == 1,
              "int(double,char) must match __cdecl");

// Variadic __cdecl keeps its own specialization, not the device default.
static_assert(classify_fn<int __cdecl(int, ...)>::value == 3,
              "variadic __cdecl must match its own specialization");

// __vectorcall must not be folded into the device default alongside __cdecl.
static_assert(classify_fn<void __vectorcall()>::value == 2,
              "__vectorcall must match its own specialization, not __cdecl");

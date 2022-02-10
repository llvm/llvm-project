// RUN: %clang_cc1 -std=c++2b -triple x86_64-linux -Wpre-c++2b-compat -fsyntax-only -verify=cxx2b %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-linux -fsyntax-only -verify=cxx20 %s
// RUN: %clang_cc1 -std=c++2b -triple i686-linux -fsyntax-only -verify=cxx2b-32 %s
// RUN: %clang_cc1 -x c -std=c11 -fsyntax-only -verify=c11 %s

#ifdef __cplusplus

typedef __SIZE_TYPE__ size_t;
// Assume ptrdiff_t is the signed integer type corresponding to size_t.
typedef __PTRDIFF_TYPE__ ssize_t;

template <typename, typename>
struct is_same { static constexpr bool value = false; };

template <typename T>
struct is_same<T, T> { static constexpr bool value = true; };

void SSizeT(void) {
  auto a1 = 1z;
  // cxx2b-warning@-1 {{'size_t' suffix for literals is incompatible with C++ standards before C++2b}}
  // cxx20-warning@-2 {{'size_t' suffix for literals is a C++2b extension}}
  static_assert(is_same<decltype(a1), ssize_t>::value);

  auto a2 = 1Z;
  // cxx2b-warning@-1 {{'size_t' suffix for literals is incompatible with C++ standards before C++2b}}
  // cxx20-warning@-2 {{'size_t' suffix for literals is a C++2b extension}}
  static_assert(is_same<decltype(a2), ssize_t>::value);
}

void SizeT(void) {
  auto a1 = 1uz;
  // cxx2b-warning@-1 {{'size_t' suffix for literals is incompatible with C++ standards before C++2b}}
  // cxx20-warning@-2 {{'size_t' suffix for literals is a C++2b extension}}
  static_assert(is_same<decltype(a1), size_t>::value);

  auto a2 = 1uZ;
  // cxx2b-warning@-1 {{'size_t' suffix for literals is incompatible with C++ standards before C++2b}}
  // cxx20-warning@-2 {{'size_t' suffix for literals is a C++2b extension}}
  static_assert(is_same<decltype(a2), size_t>::value);

  auto a3 = 1Uz;
  // cxx2b-warning@-1 {{'size_t' suffix for literals is incompatible with C++ standards before C++2b}}
  // cxx20-warning@-2 {{'size_t' suffix for literals is a C++2b extension}}
  static_assert(is_same<decltype(a3), size_t>::value);

  auto a4 = 1UZ;
  // cxx2b-warning@-1 {{'size_t' suffix for literals is incompatible with C++ standards before C++2b}}
  // cxx20-warning@-2 {{'size_t' suffix for literals is a C++2b extension}}
  static_assert(is_same<decltype(a4), size_t>::value);

  auto a5 = 1zu;
  // cxx2b-warning@-1 {{'size_t' suffix for literals is incompatible with C++ standards before C++2b}}
  // cxx20-warning@-2 {{'size_t' suffix for literals is a C++2b extension}}
  static_assert(is_same<decltype(a5), size_t>::value);

  auto a6 = 1Zu;
  // cxx2b-warning@-1 {{'size_t' suffix for literals is incompatible with C++ standards before C++2b}}
  // cxx20-warning@-2 {{'size_t' suffix for literals is a C++2b extension}}
  static_assert(is_same<decltype(a6), size_t>::value);

  auto a7 = 1zU;
  // cxx2b-warning@-1 {{'size_t' suffix for literals is incompatible with C++ standards before C++2b}}
  // cxx20-warning@-2 {{'size_t' suffix for literals is a C++2b extension}}
  static_assert(is_same<decltype(a7), size_t>::value);

  auto a8 = 1ZU;
  // cxx2b-warning@-1 {{'size_t' suffix for literals is incompatible with C++ standards before C++2b}}
  // cxx20-warning@-2 {{'size_t' suffix for literals is a C++2b extension}}
  static_assert(is_same<decltype(a8), size_t>::value);
}

void oor(void) {
#if __i386__
  (void)3'000'000'000z; // cxx2b-32-error {{signed 'size_t' literal is out of range of possible signed 'size_t' values}}
  (void)3'000'000'000uz;
  (void)5'000'000'000uz; // cxx2b-32-error {{'size_t' literal is out of range of possible 'size_t' values}}

  (void)0x80000000z;
  (void)0x80000000uz;
  (void)0x180000000uz; //cxx2b-32-error {{'size_t' literal is out of range of possible 'size_t' values}}
#endif
}

#else

void f(void) {
  (void)1z;  // c11-error {{'size_t' suffix for literals is a C++2b feature}}
  (void)1Z;  // c11-error {{'size_t' suffix for literals is a C++2b feature}}
  (void)1uz; // c11-error {{'size_t' suffix for literals is a C++2b feature}}
  (void)1uZ; // c11-error {{'size_t' suffix for literals is a C++2b feature}}
  (void)1Uz; // c11-error {{'size_t' suffix for literals is a C++2b feature}}
  (void)1UZ; // c11-error {{'size_t' suffix for literals is a C++2b feature}}
  (void)1zu; // c11-error {{'size_t' suffix for literals is a C++2b feature}}
  (void)1Zu; // c11-error {{'size_t' suffix for literals is a C++2b feature}}
  (void)1zU; // c11-error {{'size_t' suffix for literals is a C++2b feature}}
  (void)1ZU; // c11-error {{'size_t' suffix for literals is a C++2b feature}}
}

#endif

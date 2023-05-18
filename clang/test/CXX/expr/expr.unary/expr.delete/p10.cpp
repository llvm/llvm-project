// RUN: %clang_cc1 -std=c++1z -verify %s

using size_t = decltype(sizeof(0));
namespace std { enum class align_val_t : size_t {}; }

// Aligned version is preferred over unaligned version,
// unsized version is preferred over sized version.
template<unsigned Align>
struct alignas(Align) A {
  void operator delete(void*);
  void operator delete(void*, std::align_val_t) = delete; // expected-note {{here}}

  void operator delete(void*, size_t) = delete;
  void operator delete(void*, size_t, std::align_val_t) = delete;
};
void f(A<__STDCPP_DEFAULT_NEW_ALIGNMENT__> *p) { delete p; }
void f(A<__STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2> *p) { delete p; } // expected-error {{deleted}}

template<unsigned Align>
struct alignas(Align) B {
  void operator delete(void*, size_t);
  void operator delete(void*, size_t, std::align_val_t) = delete; // expected-note {{here}}
};
void f(B<__STDCPP_DEFAULT_NEW_ALIGNMENT__> *p) { delete p; }
void f(B<__STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2> *p) { delete p; } // expected-error {{deleted}}

// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -std=c++17 -fsycl-is-host -verify %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -std=c++17 -fsycl-is-device -verify %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -std=c++20 -fsycl-is-host -verify %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -std=c++20 -fsycl-is-device -verify %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -std=c++23 -fsycl-is-host -verify %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -std=c++23 -fsycl-is-device -verify %s

// These tests validate diagnostics for invalid use of 'this' in the body of
// a function declared with the sycl_kernel_entry_point attribute.


template<typename T> struct remove_reference_t {
  using type = T;
};
template<typename T> struct remove_reference_t<T&> {
  using type = T;
};

namespace std {
struct type_info {
  virtual ~type_info();
};
} // namespace std

// A generic kernel launch function.
template<typename KernelName, typename... Ts>
void sycl_kernel_launch(const char *, Ts...) {}

////////////////////////////////////////////////////////////////////////////////
// Valid declarations.
////////////////////////////////////////////////////////////////////////////////
template<int, int=0> struct KN;

struct S1 {
  [[clang::sycl_kernel_entry_point(KN<1>)]] void ok1() {
    (void)sizeof(this);
  }
};

struct S2 {
  [[clang::sycl_kernel_entry_point(KN<2>)]] void ok2() {
    (void)noexcept(this);
  }
};

struct S3 {
  [[clang::sycl_kernel_entry_point(KN<3>)]] void ok3() {
    decltype(this) x = nullptr;
  }
};

struct S4 {
  static void smf();
  [[clang::sycl_kernel_entry_point(KN<4>)]] void ok4() {
    remove_reference_t<decltype(*this)>::type::smf();
  }
};

struct S5 {
  int dm;
  void mf();
  [[clang::sycl_kernel_entry_point(KN<5>)]] void ok5() {
    (void)typeid(*this); // S5 is not abstract, so 'this' is not evaluated.
    (void)typeid(dm);    // 'int' is not an abstract class type; implicit 'this' is not evaluated.
    (void)typeid(mf());  // 'void' is not an abstract class type; implicit 'this' is not evaluated.
  }
};

template<typename KN, bool B>
struct S6 {
  void mf() noexcept(B);
  [[clang::sycl_kernel_entry_point(KN)]] void ok6() noexcept(noexcept(mf())) {}
};
template void S6<KN<6,0>, false>::ok6();
template void S6<KN<6,1>, true>::ok6();

template<typename KN, bool B>
struct S7 {
  void mf() noexcept(B);
  [[clang::sycl_kernel_entry_point(KN)]] void ok7() noexcept(noexcept(this->mf())) {}
};
template void S7<KN<7,0>, false>::ok7();
template void S7<KN<7,1>, true>::ok7();

#if __cplusplus >= 202002L
template<typename KN, typename T>
struct S8 {
  void mf(T);
  [[clang::sycl_kernel_entry_point(KN)]] void ok8() requires(requires { mf(1); }) {}
};
template void S8<KN<8>, int>::ok8();

template<typename KN, typename T>
struct S9 {
  void mf(T);
  [[clang::sycl_kernel_entry_point(KN)]] void ok9() requires(requires { this->mf(1); }) {}
};
template void S9<KN<9>, int>::ok9();
#endif


////////////////////////////////////////////////////////////////////////////////
// Invalid declarations.
////////////////////////////////////////////////////////////////////////////////

template<int, int=0> struct BADKN;

// expected-error@+3 {{'this' cannot be used in a potentially evaluated expression in the body of a function declared with the 'clang::sycl_kernel_entry_point' attribute}}
struct B1 {
  [[clang::sycl_kernel_entry_point(BADKN<1>)]] void bad1() {
    (void)this;
  }
};

// expected-error@+4 {{'this' cannot be implicitly used in a potentially evaluated expression in the body of a function declared with the 'clang::sycl_kernel_entry_point' attribute}}
struct B2 {
  int dm;
  [[clang::sycl_kernel_entry_point(BADKN<2>)]] void bad2() {
    (void)dm;
  }
};

// expected-error@+4 {{'this' cannot be implicitly used in a potentially evaluated expression in the body of a function declared with the 'clang::sycl_kernel_entry_point' attribute}}
struct B3 {
  void mf();
  [[clang::sycl_kernel_entry_point(BADKN<3>)]] void bad3() {
    (void)mf();
  }
};

// expected-error@+4 {{'this' cannot be used in a potentially evaluated expression in the body of a function declared with the 'clang::sycl_kernel_entry_point' attribute}}
struct B4 {
  virtual void vmf() = 0;
  [[clang::sycl_kernel_entry_point(BADKN<4>)]] void bad4() {
    (void)typeid(*this); // B4 is abstract, so 'this' is evaluated.
  }
};

// A diagnostic is not currently issued for uninstantiated definitions. In this
// case, a declaration is instantiated, but a definition isn't. A diagnostic
// will be issued if a definition is instantiated (as the next test exercises).
struct B5 {
  template<typename KN>
  [[clang::sycl_kernel_entry_point(KN)]] void bad5() {
    (void)this;
  }
};
extern template void B5::bad5<BADKN<5>>();

// expected-error@+4 {{'this' cannot be used in a potentially evaluated expression in the body of a function declared with the 'clang::sycl_kernel_entry_point' attribute}}
struct B6 {
  template<typename KN>
  [[clang::sycl_kernel_entry_point(KN)]] void bad6() {
    (void)this;
  }
};
// expected-note@+1 {{in instantiation of function template specialization 'B6::bad6<BADKN<6>>' requested here}}
template void B6::bad6<BADKN<6>>();

// A diagnostic is not currently issued for uninstantiated definitions. In this
// case, a declaration is instantiated, but a definition isn't. A diagnostic
// will be issued if a definition is instantiated (as the next test exercises).
template<typename KN>
struct B7 {
  [[clang::sycl_kernel_entry_point(KN)]] void bad7() {
    (void)this;
  }
};
extern template void B7<BADKN<7>>::bad7();

// expected-error@+4 {{'this' cannot be used in a potentially evaluated expression in the body of a function declared with the 'clang::sycl_kernel_entry_point' attribute}}
template<typename KN>
struct B8 {
  [[clang::sycl_kernel_entry_point(KN)]] void bad8() {
    (void)this;
  }
};
// expected-note@+1 {{in instantiation of member function 'B8<BADKN<8>>::bad8' requested here}}
template void B8<BADKN<8>>::bad8();

#if __cplusplus >= 202302L
struct B9 {
  // expected-error@+1 {{the 'clang::sycl_kernel_entry_point' attribute cannot be applied to a function with an explicit object parameter}}
  [[clang::sycl_kernel_entry_point(BADKN<9>)]] void bad9(this B9 self) {
    (void)self;
  }
};
#endif

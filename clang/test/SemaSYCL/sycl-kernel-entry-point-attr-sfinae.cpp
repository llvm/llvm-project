// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fsyntax-only -fsycl-is-device -verify %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++20 -fsyntax-only -fsycl-is-device -verify %s

// These tests are intended to validate that a sycl_kernel_entry_point attribute
// appearing in the declaration of a function template does not affect overload
// resolution or cause spurious errors during overload resolution due to either
// a substitution failure in the attribute argument or a semantic check of the
// attribute during instantiation of a specialization unless that specialization
// is selected by overload resolution.

// FIXME: C++23 [temp.expl.spec]p12 states:
// FIXME:   ... Similarly, attributes appearing in the declaration of a template
// FIXME:   have no effect on an explicit specialization of that template.
// FIXME: Clang currently instantiates and propagates attributes from a function
// FIXME: template to its explicit specializations resulting in the following
// FIXME: spurious error.
struct S1; // #S1-decl
// expected-error@+4 {{incomplete type 'S1' named in nested name specifier}}
// expected-note@+5 {{in instantiation of function template specialization 'ok1<S1>' requested here}}
// expected-note@#S1-decl {{forward declaration of 'S1'}}
template<typename T>
[[clang::sycl_kernel_entry_point(typename T::invalid)]] void ok1() {}
template<>
void ok1<S1>() {}
void test_ok1() {
  // ok1<S1>() is not a call to a SYCL kernel entry point function.
  ok1<S1>();
}

// FIXME: The sycl_kernel_entry_point attribute should not be instantiated
// FIXME: until after overload resolution has completed.
struct S2; // #S2-decl
// expected-error@+6 {{incomplete type 'S2' named in nested name specifier}}
// expected-note@+10 {{in instantiation of function template specialization 'ok2<S2>' requested here}}
// expected-note@#S2-decl {{forward declaration of 'S2'}}
template<typename T>
[[clang::sycl_kernel_entry_point(T)]] void ok2(int) {}
template<typename T>
[[clang::sycl_kernel_entry_point(typename T::invalid)]] void ok2(long) {}
void test_ok2() {
  // ok2(int) is a better match and is therefore selected by overload
  // resolution; the attempted instantiation of ok2(long) should not produce
  // an error for the substitution failure into the attribute argument.
  ok2<S2>(2);
}

// FIXME: The sycl_kernel_entry_point attribute should not be instantiated
// FIXME: until after overload resolution has completed.
struct S3;
struct Select3 {
  using bad_type = int;
  using good_type = S3;
};
// expected-error@+5 {{'typename Select3::bad_type' (aka 'int') is not a valid SYCL kernel name type; a non-union class type is required}}
// expected-note@+9 {{in instantiation of function template specialization 'ok3<Select3>' requested here}}
template<typename T>
[[clang::sycl_kernel_entry_point(typename T::good_type)]] void ok3(int) {}
template<typename T>
[[clang::sycl_kernel_entry_point(typename T::bad_type)]] void ok3(long) {}
void test_ok3() {
  // ok3(int) is a better match and is therefore selected by overload
  // resolution; the attempted instantiation of ok3(long) should not produce
  // an error for the invalid kernel name provided as the attribute argument.
  ok3<Select3>(2);
}

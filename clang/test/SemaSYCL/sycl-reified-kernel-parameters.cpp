// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fsyntax-only -fsycl-is-host -fcxx-exceptions -verify %s
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -std=c++17 -fsyntax-only -fsycl-is-device -verify %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++20 -fsyntax-only -fsycl-is-host -fcxx-exceptions -verify %s
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -std=c++20 -fsyntax-only -fsycl-is-device -verify %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++23 -fsyntax-only -fsycl-is-host -fcxx-exceptions -verify %s
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -std=c++23 -fsyntax-only -fsycl-is-device -verify %s

// Test overload resolution for implicit calls to sycl_kernel_launch<KI>(...)
// and sycl_reify_kernel_parameters<KI>(...) synthesized for functions declared
// with the sycl_kernel_entry_point attribute which have a parameter type that
// is, or has a subobject type, declared with the sycl_reified_kernel_parameter
// attribute.

////////////////////////////////////////////////////////////////////////////////
// Valid declarations.
////////////////////////////////////////////////////////////////////////////////

// A unique kernel name type is required for each declared kernel entry point.
template<int, int = 0> struct KN;

// A generic sycl_reified_kernel_parameter attributed class type.
struct [[clang::sycl_reified_kernel_parameter]] SRKP {};

// A generic kernel object type.
template<int, int = 0>
struct KT {
  void operator()() const;
  SRKP srkp;
};


// sycl_reify_kernel_parameters as function template at namespace scope.
namespace ok1 {
  template<typename KI, typename... Ts>
  auto sycl_kernel_launch(Ts&... args) {
    return [&] (auto&... subobjects) {};
  }
  template<typename KI, typename... Ts>
  auto sycl_reify_kernel_parameters(Ts&... args) {
    return [&] (auto&... subobjects) {};
  }
  [[clang::sycl_kernel_entry_point(KN<1>)]]
  void skep(KT<1> k) {
    k();
  }
}


////////////////////////////////////////////////////////////////////////////////
// Invalid declarations.
////////////////////////////////////////////////////////////////////////////////

// A unique kernel name type is required for each declared kernel entry point.
template<int, int = 0> struct BADKN;


// A generic kernel object type.
template<int, int = 0>
struct BADKT {
  void operator()() const;
  SRKP srkp;
};


// Undeclared sycl_reify_kernel_parameters identifier.
namespace bad1 {
  template<typename KI, typename... Ts>
  auto sycl_kernel_launch(Ts&... args) {
    return [&] (auto&... subobjects) {};
  }
  // expected-error@+4 {{use of undeclared identifier 'sycl_reify_kernel_parameters'}}
  // expected-note@+2  {{this indicates a problem with the SYCL runtime header files; please consider reporting this to your SYCL runtime provider}}
  // expected-note@+1  {{in implicit call to 'sycl_reify_kernel_parameters' with template argument '__sycl_kernel_info<BADKN<1>>' and function arguments (lvalue of type 'BADKT<1>') required here}}
  [[clang::sycl_kernel_entry_point(BADKN<1>)]]
  void skep(BADKT<1> k) {
    k();
  }
}

// sycl_kernel_launch returns void when a callable object is required.
namespace bad2 {
  template<typename KI, typename... Ts>
  void sycl_kernel_launch(Ts&... args) {}
  template<typename KI, typename... Ts>
  auto sycl_reify_kernel_parameters(Ts&... args) {
    return [&] (auto&... subobjects) {};
  }
  // expected-error@+4 {{called object type 'void' is not a function or function pointer}}
  // expected-note@+2  {{this indicates a problem with the SYCL runtime header files; please consider reporting this to your SYCL runtime provider}}
  // expected-note@+1  {{in implicit call to the callable returned by 'sycl_kernel_launch' with function arguments (lvalue of type 'SRKP') required here}}
  [[clang::sycl_kernel_entry_point(BADKN<2>)]]
  void skep(BADKT<2> k) {
    k();
  }
}

// sycl_reify_kernel_parameters returns void, but a callable object is required.
namespace bad3 {
  template<typename KI, typename... Ts>
  auto sycl_kernel_launch(Ts&... args) {
    return [&] (auto&... subobjects) {};
  }
  template<typename KI, typename... Ts>
  void sycl_reify_kernel_parameters(Ts&... args) {}
  // expected-error@+4 {{called object type 'void' is not a function or function pointer}}
  // expected-note@+2  {{this indicates a problem with the SYCL runtime header files; please consider reporting this to your SYCL runtime provider}}
  // expected-note@+1  {{in implicit call to the callable returned by 'sycl_reify_kernel_parameters' with function arguments (lvalue of type 'SRKP') required here}}
  [[clang::sycl_kernel_entry_point(BADKN<3>)]]
  void skep(BADKT<3> k) {
    k();
  }
}

// sycl_kernel_launch returns non-void but does not return a callable object.
namespace bad4 {
  template<typename KI, typename... Ts>
  auto sycl_kernel_launch(Ts&... args) {
    return nullptr;
  }
  template<typename KI, typename... Ts>
  auto sycl_reify_kernel_parameters(Ts&... args) {
    return [&] (auto&... subobjects) {};
  }
  // expected-error@+4 {{called object type 'std::nullptr_t' is not a function or function pointer}}
  // expected-note@+2  {{this indicates a problem with the SYCL runtime header files; please consider reporting this to your SYCL runtime provider}}
  // expected-note@+1  {{in implicit call to the callable returned by 'sycl_kernel_launch' with function arguments (lvalue of type 'SRKP') required here}}
  [[clang::sycl_kernel_entry_point(BADKN<4>)]]
  void skep(BADKT<4> k) {
    k();
  }
}

// sycl_reify_kernel_parameters returns non-void but does not return a callable object.
namespace bad5 {
  template<typename KI, typename... Ts>
  auto sycl_kernel_launch(Ts&... args) {
    return [&] (auto&... subobjects) {};
  }
  template<typename KI, typename... Ts>
  auto sycl_reify_kernel_parameters(Ts&... args) {
    return 5;
  }
  // expected-error@+4 {{called object type 'int' is not a function or function pointer}}
  // expected-note@+2  {{this indicates a problem with the SYCL runtime header files; please consider reporting this to your SYCL runtime provider}}
  // expected-note@+1  {{in implicit call to the callable returned by 'sycl_reify_kernel_parameters' with function arguments (lvalue of type 'SRKP') required here}}
  [[clang::sycl_kernel_entry_point(BADKN<5>)]]
  void skep(BADKT<5> k) {
    k();
  }
}

// sycl_kernel_launch returns a callable that does not return void.
namespace bad6 {
  template<typename KI, typename... Ts>
  auto sycl_kernel_launch(Ts&... args) {
    return [&] (auto&... subobjects) {
      return 6;
    };
  }
  template<typename KI, typename... Ts>
  auto sycl_reify_kernel_parameters(Ts&... args) {
    return [&] (auto&... subobjects) {};
  }
  // FIXME: not yet diagnosed.
  [[clang::sycl_kernel_entry_point(BADKN<6>)]]
  void skep(BADKT<6> k) {
    k();
  }
}

// sycl_reify_kernel_parameters returns a callable that does not return void.
namespace bad7 {
  template<typename KI, typename... Ts>
  auto sycl_kernel_launch(Ts&... args) {
    return [&] (auto&... subobjects) {};
  }
  template<typename KI, typename... Ts>
  auto sycl_reify_kernel_parameters(Ts&... args) {
    return [&] (auto&... subobjects) {
      return 7;
    };
  }
  // FIXME: not yet diagnosed.
  [[clang::sycl_kernel_entry_point(BADKN<7>)]]
  void skep(BADKT<7> k) {
    k();
  }
}


// FIXME: Unlike sycl_kernel_launch(), sycl_reify_kernel_parameters() cannot be
// FIXME: a non-static member function since 'this' is not available. Can it be
// FIXME: a namespace scope (const) variable template?

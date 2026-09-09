// RUN: %clang_cc1 -fsyntax-only -std=c++14 -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++17 -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++20 -verify %s


namespace UndeducedDeletedFunction {
auto deleted_auto() = delete; // #deleted_auto_decl
// expected-note@#deleted_auto_decl {{candidate function has been explicitly deleted}}
// expected-note@#deleted_auto_decl {{'deleted_auto' declared here}}

auto auto_test() {
  auto x = deleted_auto(); // #auto_use
  // expected-error@#auto_use {{call to deleted function 'deleted_auto'}}
  // expected-note@#deleted_auto_decl {{candidate function has been explicitly deleted}}
  // expected-error@#auto_use {{function 'deleted_auto' with deduced return type cannot be used before it is defined}}
  // expected-note@#deleted_auto_decl {{'deleted_auto' declared here}}
  return x;
}

auto decltype_auto_test() {
  decltype(auto) x = deleted_auto(); // #decl_type_use
  // expected-error@#decl_type_use {{call to deleted function 'deleted_auto'}}
  // expected-note@#deleted_auto_decl {{candidate function has been explicitly deleted}}
  // expected-error@#decl_type_use {{function 'deleted_auto' with deduced return type cannot be used before it is defined}}
  // expected-note@#deleted_auto_decl {{'deleted_auto' declared here}}
  return x;
}

auto decltype_auto_type_test() {
  __auto_type x = deleted_auto(); // #__auto_type_use
  // expected-error@#__auto_type_use {{call to deleted function 'deleted_auto'}}
  // expected-error@#__auto_type_use {{function 'deleted_auto' with deduced return type cannot be used before it is defined}}
  return x;
}
}

namespace DeletedWithUnresolvedExceptionSpec {
struct X {};

template <typename T> void failed_exception_spec(T) noexcept(T::value) = delete; // #failed_exception_spec_declared
// expected-error@#failed_exception_spec_declared {{no member named 'value' in 'DeletedWithUnresolvedExceptionSpec::X'}}
// expected-note@#failed_exception_used {{in instantiation of exception specification for 'failed_exception_spec<DeletedWithUnresolvedExceptionSpec::X>' requested here}}

void g() {
  failed_exception_spec(X{}); // #failed_exception_used
  // expected-error@#failed_exception_used {{call to deleted function 'failed_exception_spec'}}
  // expected-note@#failed_exception_spec_declared {{candidate function [with T = DeletedWithUnresolvedExceptionSpec::X] has been explicitly deleted}}
}
}

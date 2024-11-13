// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++23 -fexperimental-cxx-type-aware-allocators -fexceptions

namespace std {
  template <class T> struct type_identity {};
  enum class align_val_t : __SIZE_TYPE__ {};
  struct destroying_delete_t { explicit destroying_delete_t() = default; };
}

using size_t = __SIZE_TYPE__;

struct S1 {
  constexpr explicit S1() : i(5) {  }
  const int i;
};

void *operator new(std::type_identity<S1>, size_t sz); // #1
void operator delete(std::type_identity<S1>, void* ptr); // #2

constexpr int ensure_consteval_skips_typed_allocators() {
  // Verify we dont resolve typed allocators in const contexts
  auto * s = new S1();
  auto result = s->i;
  delete s;
  return result;
};

struct S2 {
  constexpr explicit S2() : i(5) {  }
  const int i;
};

void *operator new(std::type_identity<S2>, size_t sz) = delete; // #3
void operator delete(std::type_identity<S2>, void* ptr) = delete; // #4

constexpr int ensure_constexpr_retains_types_at_runtime() {
  // Verify we dont resolve typed allocators in const contexts
  S2 *s = new S2();
  // expected-error@-1 {{call to deleted function 'operator new'}}
  // expected-note@#1 {{candidate function not viable: no known conversion from 'type_identity<S2>' to 'type_identity<S1>' for 1st argument}}
  // expected-note@#3 {{candidate function has been explicitly deleted}}
  auto result = s->i;
  delete s;
  // expected-error@-1 {{attempt to use a deleted function}}
  // expected-note@#4 {{'operator delete' has been explicitly marked deleted here}}
  return result;
};


struct S3 {
  constexpr explicit S3() : i(5) {  }
  const int i;
  template <typename T> void* operator new(std::type_identity<T>, size_t sz) = delete; // #5
  template <typename T> void operator delete(std::type_identity<T>, void *) = delete; // #6
};

template <typename T> void* operator new(std::type_identity<T>, size_t sz) = delete; // #7
template <typename T> void operator delete(std::type_identity<T>, void *) = delete; // #8

constexpr int constexpr_vs_inclass_operators() {
  S3 *s;
  if consteval {
    s = ::new S3();
  } else {
    s = new S3();
    // expected-error@-1 {{call to deleted function 'operator new'}}
    // expected-note@#5 {{candidate function [with T = S3] has been explicitly deleted}}
  }
  auto result = s->i;
  if consteval {
    ::delete s;
  } else {
    delete s;
    // expected-error@-1 {{attempt to use a deleted function}}
    // expected-note@#6 {{'operator delete<S3>' has been explicitly marked deleted here}}
  }
  return result;
};

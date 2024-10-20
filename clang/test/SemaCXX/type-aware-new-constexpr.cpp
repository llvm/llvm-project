// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++2c -fexperimental-cxx-type-aware-allocators -fexceptions 

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

void *operator new(std::type_identity<S1>, size_t sz);
// expected-note@-1 {{candidate function not viable: no known conversion from 'type_identity<S2>' to 'type_identity<S1>' for 1st argument}}
void operator delete(std::type_identity<S1>, void* ptr);

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

void *operator new(std::type_identity<S2>, size_t sz) = delete;
// expected-note@-1 {{candidate function has been explicitly deleted}}
void operator delete(std::type_identity<S2>, void* ptr) = delete;
// expected-note@-1 {{'operator delete' has been explicitly marked deleted here}}

constexpr int ensure_constexpr_retains_types_at_runtime() {
  // Verify we dont resolve typed allocators in const contexts
  S2 *s = new S2(); // expected-error {{call to deleted function 'operator new'}}
  auto result = s->i;
  delete s; // expected-error {{attempt to use a deleted function}}
  return result;
};


struct S3 {
  constexpr explicit S3() : i(5) {  }
  const int i;
  template <typename T> void* operator new(std::type_identity<T>, size_t sz) = delete;
  // expected-note@-1 {{candidate function [with T = S3] has been explicitly deleted}}
  template <typename T> void operator delete(std::type_identity<T>, void *) = delete;
  // expected-note@-1 {{'operator delete<S3>' has been explicitly marked deleted here}}
};

template <typename T> void* operator new(std::type_identity<T>, size_t sz) = delete;
template <typename T> void operator delete(std::type_identity<T>, void *) = delete;

constexpr int constexpr_vs_inclass_operators() {
  S3 *s;
  if consteval {
    s = ::new S3();
  } else {
    s = new S3(); // expected-error {{call to deleted function 'operator new'}}
  }
  auto result = s->i;
  if consteval {
    ::delete s;
  } else {
    delete s; // expected-error {{attempt to use a deleted function}}
  }
  return result;
};

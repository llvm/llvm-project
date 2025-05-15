// RUN: %clang_cc1 -fsyntax-only -verify %s        -std=c++26 -fexceptions -fcxx-exceptions    -fsized-deallocation    -faligned-allocation
// RUN: %clang_cc1 -fsyntax-only -verify %s        -std=c++26 -fexceptions -fcxx-exceptions -fno-sized-deallocation    -faligned-allocation
// RUN: %clang_cc1 -fsyntax-only -verify %s        -std=c++26 -fexceptions -fcxx-exceptions -fno-sized-deallocation -fno-aligned-allocation
// RUN: %clang_cc1 -fsyntax-only -verify %s        -std=c++26 -fexceptions -fcxx-exceptions    -fsized-deallocation -fno-aligned-allocation

namespace std {
  template <class T> struct type_identity {};
  enum class align_val_t : __SIZE_TYPE__ {};
  struct destroying_delete_t { explicit destroying_delete_t() = default; };
}

using size_t = __SIZE_TYPE__;
struct Context;
struct S1 {
  S1();
};
void *operator new(std::type_identity<S1>, size_t, std::align_val_t, Context&);
void operator delete(std::type_identity<S1>, void*, size_t, std::align_val_t, Context&) = delete; // #1

struct S2 {
  S2();
  template<typename T> void *operator new(std::type_identity<T>, size_t, std::align_val_t, Context&);
  template<typename T> void operator delete(std::type_identity<T>, void*, size_t, std::align_val_t, Context&) = delete; // #2
  template<typename T> void operator delete(std::type_identity<T>, void*, size_t, std::align_val_t) = delete; // #3
};

struct S3 {
  S3();
  template<typename T> void *operator new(std::type_identity<T>, size_t, std::align_val_t);
  template<typename T> void operator delete(std::type_identity<T>, void*, size_t, std::align_val_t) = delete; // #4
};

struct S4 {
  S4();
  template<typename T> void *operator new(std::type_identity<T>, size_t, std::align_val_t, Context&); // #S4_new
  template<typename T> void operator delete(std::type_identity<T>, void*, size_t, std::align_val_t); // #5
};

void test(Context& Ctx) {
  S1 *s1 = new (Ctx) S1;
  // expected-error@-1 {{attempt to use a deleted function}}
  // expected-note@#1 {{'operator delete' has been explicitly marked deleted here}}
  delete s1;
  S2 *s2_1 = new (Ctx) S2;
  // expected-error@-1 {{attempt to use a deleted function}}
  // expected-note@#2 {{'operator delete<S2>' has been explicitly marked deleted here}}
  // expected-note@#3 {{'operator delete<S2>' has been explicitly marked deleted here}}
  delete s2_1;
  // expected-error@-1 {{attempt to use a deleted function}}
  // expected-note@#2 {{'operator delete<S2>' has been explicitly marked deleted here}}
  S2 *s2_2 = new (std::align_val_t(128), Ctx) S2;
  // expected-error@-1 {{attempt to use a deleted function}}
  delete s2_2;
  // expected-error@-1 {{attempt to use a deleted function}}
  // expected-note@#3 {{'operator delete<S2>' has been explicitly marked deleted here}}
  S3 *s3_1 = new S3;
  // expected-error@-1 {{attempt to use a deleted function}}
  // expected-note@#4 {{'operator delete<S3>' has been explicitly marked deleted here}}
  delete s3_1;
  // expected-error@-1 {{attempt to use a deleted function}}
  // expected-note@#4 {{'operator delete<S3>' has been explicitly marked deleted here}}
  S3 *s3_2 = new (std::align_val_t(128)) S3;
  // expected-error@-1 {{attempt to use a deleted function}}
  // expected-note@#4 {{'operator delete<S3>' has been explicitly marked deleted here}}
  delete s3_2;
  // expected-error@-1 {{attempt to use a deleted function}}
  // expected-note@#4 {{'operator delete<S3>' has been explicitly marked deleted here}}

  S4 *s4_1 = new (Ctx) S4;
  // expected-error@-1 {{type aware 'operator new' requires a matching type aware placement 'operator delete' to be declared in the same scope}}
  // expected-note@#S4_new {{type aware 'operator new' declared here in 'S4'}}
  delete s4_1;
}

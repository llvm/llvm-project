// RUN: %clang_cc1 -fsyntax-only -verify %s        -std=c++23 -fexperimental-cxx-type-aware-allocators -fexceptions

namespace std {
  template <class T> struct type_identity {};
  enum class align_val_t : __SIZE_TYPE__ {};
  struct destroying_delete_t { explicit destroying_delete_t() = default; };
}

static_assert(__has_feature(cxx_type_aware_allocators));
#ifdef TADD
static_assert(__has_feature(cxx_type_aware_destroying_delete));
#else
static_assert(!__has_feature(cxx_type_aware_destroying_delete));
#endif

using size_t = __SIZE_TYPE__;
struct Context;
struct S1 {
  S1() throw();
};
void *operator new(std::type_identity<S1>, size_t, Context&);
void operator delete(std::type_identity<S1>, void*, Context&) = delete; // #1

struct S2 {
  S2() throw();
  template<typename T> void *operator new(std::type_identity<T>, size_t, Context&);
  template<typename T> void operator delete(std::type_identity<T>, void*, Context&) = delete; // #2
};

void test(Context& Ctx) {
  S1 *s1 = new (Ctx) S1;
  // expected-error@-1 {{attempt to use a deleted function}}
  // expected-note@#1 {{'operator delete' has been explicitly marked deleted here}}
  delete s1;
  S2 *s2 = new (Ctx) S2;
  // expected-error@-1 {{attempt to use a deleted function}}
  // expected-note@#2 {{'operator delete<S2>' has been explicitly marked deleted here}}
  delete s2;
  // expected-error@-1 {{no suitable member 'operator delete' in 'S2'}}
  // expected-note@#2 {{member 'operator delete' declared here}}
}

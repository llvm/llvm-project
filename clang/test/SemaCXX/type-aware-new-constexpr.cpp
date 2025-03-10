// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++23 -fexperimental-cxx-type-aware-allocators -fexceptions                                             -fsized-deallocation    -faligned-allocation 
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++23 -fexperimental-cxx-type-aware-allocators -fexceptions                                          -fno-sized-deallocation    -faligned-allocation 
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++23 -fexperimental-cxx-type-aware-allocators -fexceptions                                             -fsized-deallocation -fno-aligned-allocation 
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++23 -fexperimental-cxx-type-aware-allocators -fexceptions                                          -fno-sized-deallocation -fno-aligned-allocation 
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++23 -fexperimental-cxx-type-aware-allocators -fexceptions -fexperimental-new-constant-interpreter     -fsized-deallocation    -faligned-allocation
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++23 -fexperimental-cxx-type-aware-allocators -fexceptions -fexperimental-new-constant-interpreter  -fno-sized-deallocation    -faligned-allocation
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++23 -fexperimental-cxx-type-aware-allocators -fexceptions -fexperimental-new-constant-interpreter     -fsized-deallocation -fno-aligned-allocation
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++23 -fexperimental-cxx-type-aware-allocators -fexceptions -fexperimental-new-constant-interpreter  -fno-sized-deallocation -fno-aligned-allocation


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

void *operator new(std::type_identity<S1>, size_t sz, std::align_val_t); // #1
void operator delete(std::type_identity<S1>, void* ptr, size_t sz, std::align_val_t); // #2

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

void *operator new(std::type_identity<S2>, size_t sz, std::align_val_t) = delete; // #3
void operator delete(std::type_identity<S2>, void* ptr, size_t sz, std::align_val_t) = delete; // #4

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
  template <typename T> void* operator new(std::type_identity<T>, size_t sz, std::align_val_t) = delete; // #5
  template <typename T> void operator delete(std::type_identity<T>, void *, size_t sz, std::align_val_t) = delete; // #6
};

template <typename T> void* operator new(std::type_identity<T>, size_t sz, std::align_val_t) = delete; // #7
template <typename T> void operator delete(std::type_identity<T>, void *, size_t sz, std::align_val_t) = delete; // #8

constexpr int constexpr_vs_inclass_operators() {
  S3 *s;
  if consteval {
    s = ::new S3();
    // expected-error@-1 {{call to deleted function 'operator new'}}
    // expected-note@#1 {{candidate function not viable: no known conversion from 'type_identity<S3>' to 'type_identity<S1>' for 1st argument}}
    // expected-note@#3 {{candidate function not viable: no known conversion from 'type_identity<S3>' to 'type_identity<S2>' for 1st argument}}
    // expected-note@#7 {{candidate function [with T = S3] has been explicitly deleted}}
  } else {
    s = new S3();
    // expected-error@-1 {{call to deleted function 'operator new'}}
    // expected-note@#5 {{candidate function [with T = S3] has been explicitly deleted}}
  }
  auto result = s->i;
  if consteval {
    ::delete s;
    // expected-error@-1 {{attempt to use a deleted function}}
    // expected-note@#8 {{'operator delete<S3>' has been explicitly marked deleted here}}
  } else {
    delete s;
    // expected-error@-1 {{attempt to use a deleted function}}
    // expected-note@#6 {{'operator delete<S3>' has been explicitly marked deleted here}}
  }
  return result;
};

// Test a variety of valid constant evaluation paths
struct S4 {
  int i = 1;
  constexpr S4() __attribute__((noinline)) {}
};

void* operator new(std::type_identity<S4>, size_t sz, std::align_val_t);
void operator delete(std::type_identity<S4>, void *, size_t sz, std::align_val_t);

constexpr int do_dynamic_alloc(int n) {
  S4* s = new S4;
  int result = n * s->i;
  delete s;
  return result;
}

template <int N> struct Tag {
};

static constexpr int force_do_dynamic_alloc = do_dynamic_alloc(5);

constexpr int test_consteval_calling_constexpr(int i) {
  if consteval {
    return do_dynamic_alloc(2 * i);
  }
  return do_dynamic_alloc(3 * i);
}

int test_consteval(int n, Tag<test_consteval_calling_constexpr(2)>, Tag<do_dynamic_alloc(3)>) {
  static const int t1 = test_consteval_calling_constexpr(4);
  static const int t2 = do_dynamic_alloc(5);
  int t3 = test_consteval_calling_constexpr(6);
  int t4 = do_dynamic_alloc(7);
  return t1 * t2 * t3 * t4;
}

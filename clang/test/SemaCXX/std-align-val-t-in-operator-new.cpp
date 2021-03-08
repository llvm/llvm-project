// RUN: %clang_cc1 -std=c++11                      -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++11 -faligned-allocation -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++14                      -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++14 -faligned-allocation -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++17                      -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++17 -faligned-allocation -fsyntax-only -verify %s

namespace std {
typedef __SIZE_TYPE__ size_t;
struct nothrow_t {};
#if __cplusplus >= 201103L
enum class align_val_t : size_t {};
#else
enum align_val_t {
// We can't force an underlying type when targeting windows.
#ifndef _WIN32
  __zero = 0,
  __max = (size_t)-1
#endif
};
#endif
} // namespace std

void *operator new(std::size_t count, std::align_val_t al) __attribute__((alloc_align(2))); // #1

#define OVERALIGNED alignas(__STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2)

struct OVERALIGNED A {
  A();
  int n[128];
};

void *ptr_variable(int align) { return new (std::align_val_t(align)) A; }
void *ptr_align16() { return new (std::align_val_t(16)) A; }
void *ptr_align15() { return new (std::align_val_t(15)) A; } // expected-warning {{requested alignment is not a power of 2}}

struct alignas(128) S {
  S() {}
};

void *alloc_overaligned_struct() {
  return new S;
}

void *alloc_overaligned_struct_with_extra_variable_alignment(int align) {
  return new (std::align_val_t(align)) S;
}
void *alloc_overaligned_struct_with_extra_256_alignment(int align) {
  return new (std::align_val_t(256)) S;
}
void *alloc_overaligned_struct_with_extra_255_alignment(int align) {
  return new (std::align_val_t(255)) S; // expected-warning {{requested alignment is not a power of 2}}
}

std::align_val_t align_variable(int align) { return std::align_val_t(align); }
std::align_val_t align_align16() { return std::align_val_t(16); }
std::align_val_t align_align15() { return std::align_val_t(15); }

struct X {};
void *operator new(std::size_t, X); // #2
void *operator new(std::size_t, std::align_val_t, X); // #3
// FIXME: Consider improving notes 1 and 3 here to say that these are aligned
// allocation functions and the type is not over-aligned.
X *p = new (123) X; // expected-error {{no matching function}}
// expected-note@#1 {{no known conversion from 'int' to 'std::align_val_t' for 2nd argument}}
// expected-note@#2 {{no known conversion from 'int' to 'X' for 2nd argument}}
// expected-note@#3 {{requires 3 arguments}}
// expected-note@* {{requires 1 argument, but 2 were provided}} (builtin)

#ifdef __cpp_aligned_new
struct alignas(__STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2) Y {};
Y *q = new (123) Y; // expected-error {{no matching function}}
// expected-note@#1 {{requires 2 arguments, but 3 were provided}}
// expected-note@#2 {{no known conversion from 'int' to 'X' for 2nd argument}}
// expected-note@#3 {{no known conversion from 'int' to 'X' for 3rd argument}}
// expected-note@* {{requires 1 argument, but 2 were provided}} (builtin)
#endif

X *r = new (std::align_val_t(32), 123) X; // expected-error {{no matching function}}
// expected-note@#1 {{requires 2 arguments, but 3 were provided}}
// expected-note@#2 {{requires 2 arguments, but 3 were provided}}
// expected-note@#3 {{no known conversion from 'int' to 'X' for 3rd argument}}
// expected-note@* {{requires 1 argument, but 3 were provided}} (builtin)

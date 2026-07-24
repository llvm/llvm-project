// The built-in stdlib nullable-return allowlist matches the real C library
// functions, which live at global scope (or in std, e.g. std::malloc from
// <cstdlib>, possibly behind a libc++ inline namespace). A user-defined
// function that merely shares the spelling in an unrelated namespace must NOT
// be treated as nullable — otherwise the allowlist produces false positives on
// code that has nothing to do with the C library.
//
// Uses -fnullability-default=nonnull so unannotated pointer returns are nonnull
// by default; only the stdlib allowlist can make a return nullable. This is
// what isolates the scope check (under =nullable everything is nullable anyway).
//
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nonnull -Wno-nullable-to-nonnull-conversion -std=c++17 %s -verify

typedef unsigned long size_t;

// Real C library at global scope (extern "C" redecl context is the TU).
extern "C" {
void *malloc(size_t);
char *getenv(const char *);
}

// libc++-style: std with an inline namespace. isStdNamespace() sees through the
// inline namespace, so this still matches.
namespace std {
inline namespace __1 {
void *calloc(size_t, size_t);
}
} // namespace std

// Arbitrary user namespace sharing stdlib names — must NOT match.
namespace user_ns {
void *malloc(size_t);
char *getenv(const char *);
} // namespace user_ns

void real_malloc_warns() {
  void *p = malloc(4);
  *((int *)p) = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

void std_inline_ns_calloc_warns() {
  void *p = std::calloc(1, 4);
  *((int *)p) = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// user_ns::malloc returns nonnull (default), is not the C library malloc, so
// no nullable-deref warning fires.
void user_namespace_malloc_silent() {
  void *p = user_ns::malloc(4);
  *((int *)p) = 1; // no warning
}

void user_namespace_getenv_silent() {
  char *p = user_ns::getenv("X");
  *p = 'a'; // no warning
}

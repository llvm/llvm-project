#include <ptrcheck.h>

// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
// RUN: not %clang_cc1 -fsyntax-only -fbounds-safety -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

void concrete_to_incomplete(int *p) {
  // expected-note@+3{{cast to 'int *__bidi_indexable' first to keep bounds of 'p'}}
  // expected-note@+2{{silence by making the destination '__single'}}
  // expected-warning@+1{{casting 'int *__single' to 'void *__bidi_indexable' creates a '__bidi_indexable' pointer with zero length due to 'void' having unknown size)}}
  void *l = p;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:13-[[@LINE-1]]:13}:"(int *__bidi_indexable)"
}

void concrete_to_incomplete_apply_fixit(int *p) {
  void *l = (int *__bidi_indexable)p;
}

// expected-note@+1{{pointer 'p' declared here}}
void concrete_to_incomplete_indi(int *p) {
  // expected-note@+4{{cast to 'int *__bidi_indexable' first to keep bounds of 'p'}}
  // expected-note@+3{{silence by making the destination '__single'}}
  // expected-warning@+2{{casting 'int *__single' to 'void *__indexable' creates a '__indexable' pointer with zero length due to 'void' having unknown size)}}
  // expected-warning@+1{{initializing type 'void *__indexable' with an expression of type 'int *__single' results in an __indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'p'}}
  void *__indexable l = p;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:25-[[@LINE-1]]:25}:"(int *__bidi_indexable)"
}

void concrete_to_incomplete_indi_apply_fixit(int *p) {
  void *__indexable l = (int *__bidi_indexable)p;
}

// expected-note@+1{{pointer 'p' declared here}}
void concrete_to_incomplete_bidi(int *p) {
  // expected-note@+4{{cast to 'int *__bidi_indexable' first to keep bounds of 'p'}}
  // expected-note@+3{{silence by making the destination '__single'}}
  // expected-warning@+2{{casting 'int *__single' to 'void *__bidi_indexable' creates a '__bidi_indexable' pointer with zero length due to 'void' having unknown size)}}
  // expected-warning@+1{{initializing type 'void *__bidi_indexable' with an expression of type 'int *__single' results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'p'}}
  void *__bidi_indexable l = p;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:30-[[@LINE-1]]:30}:"(int *__bidi_indexable)"
}

void concrete_to_incomplete_bidi_apply_fixit(int *p) {
  void *__bidi_indexable l = (int *__bidi_indexable)p;
}

void concrete_to_incomplete_explicit_bidi(int *p) {
  // expected-note@+3{{cast to 'int *__bidi_indexable' first to keep bounds of 'p'}}
  // expected-note@+2{{silence by making the destination '__single'}}
  // expected-warning@+1{{casting 'int *__single' to 'void *__bidi_indexable' creates a '__bidi_indexable' pointer with zero length due to 'void' having unknown size)}}
  void *l = (void *__bidi_indexable)p;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:37-[[@LINE-1]]:37}:"(int *__bidi_indexable)"
}

void concrete_to_incomplete_explicit_bidi_apply_fixit(int *p) {
  void *l = (void *__bidi_indexable)(int *__bidi_indexable)p;
}

void concrete_to_incomplete_explicit_indi(int *p) {
  // expected-note@+3{{cast to 'int *__bidi_indexable' first to keep bounds of 'p'}}
  // expected-note@+2{{silence by making the destination '__single'}}
  // expected-warning@+1{{casting 'int *__single' to 'void *__indexable' creates a '__indexable' pointer with zero length due to 'void' having unknown size)}}
  void *l = (void *__indexable)p;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:32-[[@LINE-1]]:32}:"(int *__bidi_indexable)"
}

void concrete_to_incomplete_explicit_indi_apply_fixit(int *p) {
  void *l = (void *__indexable)(int *__bidi_indexable)p;
}

void concrete_to_incomplete_indi_explicit(int *p) {
  // expected-note@+3{{cast to 'int *__bidi_indexable' first to keep bounds of 'p'}}
  // expected-note@+2{{silence by making the destination '__single'}}
  // expected-warning@+1{{casting 'int *__single' to 'void *__bidi_indexable' creates a '__bidi_indexable' pointer with zero length due to 'void' having unknown size)}}
  void *__indexable l = (void *__bidi_indexable)p;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:49-[[@LINE-1]]:49}:"(int *__bidi_indexable)"
}

void concrete_to_incomplete_indi_explicit_apply_fixit(int *p) {
  void *__indexable l = (void *__bidi_indexable)(int *__bidi_indexable)p;
}

void concrete_to_incomplete_bidi_explicit(int *p) {
  // expected-note@+3{{cast to 'int *__bidi_indexable' first to keep bounds of 'p'}}
  // expected-note@+2{{silence by making the destination '__single'}}
  // expected-warning@+1{{casting 'int *__single' to 'void *__bidi_indexable' creates a '__bidi_indexable' pointer with zero length due to 'void' having unknown size)}}
  void *__bidi_indexable l = (void *__bidi_indexable)p;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:54-[[@LINE-1]]:54}:"(int *__bidi_indexable)"
}

void concrete_to_incomplete_bidi_explicit_apply_fixit(int *p) {
  void *__bidi_indexable l = (void *__bidi_indexable)(int *__bidi_indexable)p;
}

void wider_to_narrower(int *p) {
  // expected-note@+3{{cast to 'int *__bidi_indexable' first to keep bounds of 'p'}}
  // expected-note@+2{{silence by making the destination '__single'}}
  // expected-warning@+1{{casting 'int *__single' to 'char *__bidi_indexable' creates a '__bidi_indexable' pointer with bounds containing only one 'char'}}
  char *l = (char *__bidi_indexable)p + 1;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:37-[[@LINE-1]]:37}:"(int *__bidi_indexable)"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:37-[[@LINE-2]]:37}:"(char *__single)"
}

void wider_to_narrower_apply_fixit1(int *p) {
  char *l = (char *__bidi_indexable)(int *__bidi_indexable)p + 1;
}

void wider_to_narrower_apply_fixit2(int *p) {
  char *l = (char *__bidi_indexable)(char *__single)p + 1;
}

// expected-note@+1{{passing argument to parameter here}}
void bidi_arg(char *__bidi_indexable);

void wider_to_narrower_arg(int *p) {
  // expected-warning@+3{{casting 'int *__single' to 'char *__bidi_indexable' creates a '__bidi_indexable' pointer with bounds containing only one 'char'}}
  // expected-note@+2{{cast to 'int *__bidi_indexable' first to keep bounds of 'p'}}
  // expected-note@+1{{silence by making the destination '__single'}}
  bidi_arg((char *__bidi_indexable)p);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:36-[[@LINE-1]]:36}:"(int *__bidi_indexable)"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:36-[[@LINE-2]]:36}:"(char *__single)"
}

void wider_to_narrower_arg_apply_fixit1(int *p) {
  bidi_arg((char *__bidi_indexable)(int *__bidi_indexable)p);
}

void wider_to_narrower_arg_apply_fixit2(int *p) {
  bidi_arg((char *__bidi_indexable)(char *__single)p);
}

void wider_to_narrower_arg2(int *p) {
  // expected-warning@+1{{passing type 'char *__single' to parameter of type 'char *__bidi_indexable' results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced}}
  bidi_arg((char *)p);
}

char *__bidi_indexable wider_to_narrower_ret(int *p) {
  // expected-warning@+3{{casting 'int *__single' to 'char *__bidi_indexable' creates a '__bidi_indexable' pointer with bounds containing only one 'char'}}
  // expected-note@+2{{cast to 'int *__bidi_indexable' first to keep bounds of 'p'}}
  // expected-note@+1{{silence by making the destination '__single'}}
  return (char *__bidi_indexable)p;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:34-[[@LINE-1]]:34}:"(int *__bidi_indexable)"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:34-[[@LINE-2]]:34}:"(char *__single)"
}

char *__bidi_indexable wider_to_narrower_ret_apply_fixit1(int *p) {
  return (char *__bidi_indexable)(int *__bidi_indexable)p;
}

char *__bidi_indexable wider_to_narrower_ret_apply_fixit2(int *p) {
  return (char *__bidi_indexable)(char *__single)p;
}

// expected-note@+1{{pointer 'p' declared here}}
void *__bidi_indexable concrete_to_incomplete_ret(int *__single p) {
  // expected-warning@+4{{casting 'int *__single' to 'void *__bidi_indexable' creates a '__bidi_indexable' pointer with zero length due to 'void' having unknown size)}}
  // expected-warning@+3{{returning type 'int *__single' from a function with result type 'void *__bidi_indexable' results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'p'}}
  // expected-note@+2{{cast to 'int *__bidi_indexable' first to keep bounds of 'p'}}
  // expected-note@+1{{silence by making the destination '__single'}}
  return p;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:10-[[@LINE-1]]:10}:"(int *__bidi_indexable)"
}

void *__bidi_indexable concrete_to_incomplete_ret_apply_fixit(int *__single p) {
  return (int *__bidi_indexable)p;
}

void narrower_to_wider(char *p) {
  int *l = (int *__bidi_indexable)p + 1;
}

void incomplete_to_concrete(void *p) {
  char *l = (char *)p;
}

void incomplete_to_concrete_bidi(void *p) {
  // expected-error@+1{{cannot cast from __single pointer to incomplete type 'void *__single' to indexable pointer type 'void *__bidi_indexable'}}
  char *l = (void *__bidi_indexable)p;
}

struct s;

void opaque_to_concrete(struct s *p) {
  // expected-error@+1{{cannot initialize indexable pointer with type 'char *__bidi_indexable' from __single pointer to incomplete type 'struct s *__single'; consider declaring pointer 'l' as '__single'}}
  char *l = p; // expected-note{{pointer 'l' declared here}}
}

// This is also undesirable because it's not obvious whether 'struct s' incomplete for the programmer.
void opaque_to_concrete_explicit(struct s *p) {
  char *l = (char *)p;
}

void opaque_to_concrete_explicit2(struct s *p) {
  // expected-error@+1{{cannot initialize indexable pointer with type 'char *__bidi_indexable' from __single pointer to incomplete type 'struct s *__single'; consider declaring pointer 'l' as '__single'}}
  char *l = (struct s *)p; // expected-note{{pointer 'l' declared here}}
}

void concrete_to_opaque(char *p) {
  // expected-note@+4{{cast to 'char *__bidi_indexable' first to keep bounds of 'p'}}
  // expected-note@+3{{silence by making the destination '__single'}}
  // expected-warning@+2{{casting 'char *__single' to 'struct s *__bidi_indexable' creates a '__bidi_indexable' pointer with zero length due to 'struct s' having unknown size)}}
  // expected-warning@+1{{incompatible pointer types initializing 'struct s *__bidi_indexable' with an expression of type 'char *__single'}}
  struct s *l = p;
}

struct t;

void opaque_to_opaque(struct t *p) {
  // expected-error@+1{{cannot initialize indexable pointer with type 'struct s *__bidi_indexable' from __single pointer to incomplete type 'struct t *__single'; consider declaring pointer 'l' as '__single'}}
  struct s *l = p; // expected-note{{pointer 'l' declared here}}
}


struct flex {
  unsigned count;
  int arr[__counted_by(count)];
};

void flexible_to_concrete_narrower(struct flex *p) {
  // expected-warning@+1{{incompatible pointer types initializing 'char *__bidi_indexable' with an expression of type 'struct flex *__single'}}
  char *l = p;
}

void flexible_to_concrete_narrower_apply_fixit1(struct flex *p) {
  // expected-warning@+1{{incompatible pointer types initializing 'char *__bidi_indexable' with an expression of type 'struct flex *__bidi_indexable'}}
  char *l = (struct flex *__bidi_indexable)p;
  }

void flexible_to_concrete_narrower_apply_fixit2(struct flex *p) {
  char *l = (char *__single)p;
}

void flexible_to_concrete_wider(struct flex *p) {
  // expected-warning@+1{{incompatible pointer types initializing 'long long *__bidi_indexable' with an expression of type 'struct flex *__single'}}
  long long *l = p;
}

void flexible_to_concrete_match(struct flex *p) {
  // expected-warning@+1{{incompatible pointer types initializing 'unsigned int *__bidi_indexable' with an expression of type 'struct flex *__single'}}
  unsigned *l = p;
}

void flexible_to_incomplete(struct flex *p) {
  void *c = p;
}

void incomplete_to_flexible(void *p) {
  // expected-error@+1{{cannot initialize indexable pointer with type 'struct flex *__bidi_indexable' from __single pointer to incomplete type 'void *__single'; consider declaring pointer 'l' as '__single'}}
  struct flex *l = p; // expected-note{{pointer 'l' declared here}}
}

void concrete_match_to_flexible(unsigned *p) {
  // expected-warning@+1{{incompatible pointer types initializing 'struct flex *__bidi_indexable' with an expression of type 'unsigned int *__single'}}
  struct flex *l = p;
}

void concrete_narrower_to_flexible(char *p) {
  // expected-warning@+1{{incompatible pointer types initializing 'struct flex *__bidi_indexable' with an expression of type 'char *__single'}}
  struct flex *l = p;
}

void concrete_wider_to_flexible(long long *p) {
  // expected-note@+4{{cast to 'long long *__bidi_indexable' first to keep bounds of 'p'}}
  // expected-note@+3{{silence by making the destination '__single'}}
  // expected-warning@+2{{casting 'long long *__single' to 'struct flex *__bidi_indexable' creates a '__bidi_indexable' pointer with bounds containing only one 'struct flex'}}
  // expected-warning@+1{{incompatible pointer types initializing 'struct flex *__bidi_indexable' with an expression of type 'long long *__single'}}
  struct flex *l = p;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:20-[[@LINE-1]]:20}:"(long long *__bidi_indexable)"
}

void concrete_wider_to_flexible_apply_fixit(long long *p) {
  // expected-warning@+1{{incompatible pointer types initializing 'struct flex *__bidi_indexable' with an expression of type 'long long *__bidi_indexable'}}
  struct flex *l = (long long *__bidi_indexable)p;
}

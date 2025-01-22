

// RUN: %clang_cc1 -fsyntax-only -verify -Wthread-safety %s
// RUN: %clang_cc1 -fbounds-safety -fsyntax-only -verify -Wthread-safety %s
// RUN: %clang_cc1 -fexperimental-bounds-safety-attributes -x c -fsyntax-only -verify -Wthread-safety %s
// RUN: %clang_cc1 -fexperimental-bounds-safety-attributes -x c++ -fsyntax-only -verify -Wthread-safety %s
// RUN: %clang_cc1 -fexperimental-bounds-safety-attributes -x objective-c -fsyntax-only -verify -Wthread-safety %s
// RUN: %clang_cc1 -fexperimental-bounds-safety-attributes -x objective-c++ -fsyntax-only -verify -Wthread-safety %s

#include <stdbool.h>

#define EXCLUSIVE_LOCK_FUNCTION(...)    __attribute__ ((exclusive_lock_function(__VA_ARGS__)))
#define SHARED_LOCK_FUNCTION(...)       __attribute__ ((shared_lock_function(__VA_ARGS__)))
#define UNLOCK_FUNCTION(...)            __attribute__ ((unlock_function(__VA_ARGS__)))

void elf_fun_params(int lvar EXCLUSIVE_LOCK_FUNCTION()); // \
  // expected-warning {{'exclusive_lock_function' attribute applies to function parameters only if their type is a reference to a 'scoped_lockable'-annotated type}}
void slf_fun_params(int lvar SHARED_LOCK_FUNCTION()); // \
  // expected-warning {{'shared_lock_function' attribute applies to function parameters only if their type is a reference to a 'scoped_lockable'-annotated type}}
void uf_fun_params(int lvar UNLOCK_FUNCTION()); // \
  // expected-warning {{'unlock_function' attribute applies to function parameters only if their type is a reference to a 'scoped_lockable'-annotated type}}

// regression tests added for rdar://92699615
typedef bool __attribute__((capability("role"))) role_t;
extern role_t guard1;
int __attribute__((guarded_by(guard1))) alloc1;
extern int guard2;
int __attribute__((guarded_by(guard2))) alloc2; // \
  // expected-warning{{'guarded_by' attribute requires arguments whose type is annotated with 'capability' attribute; type here is 'int'}}
int __attribute__((guarded_by(guard3))) alloc3; // \
// expected-error{{use of undeclared identifier 'guard3'}}

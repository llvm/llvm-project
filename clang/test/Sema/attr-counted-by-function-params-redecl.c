// RUN: %clang_cc1 -fsyntax-only -Wno-deprecated-non-prototype -verify %s
// RUN: %clang_cc1 -fexperimental-late-parse-attributes -DLATE_PARSING -fsyntax-only -Wno-deprecated-non-prototype -verify %s

// Counts are part of a function's interface, so a redeclaration has to repeat
// them, on a parameter's own pointer and on those it reaches through pointers
// alone. Each count names an earlier parameter, so it resolves with or without
// late parsing.

#define __counted_by(f)  __attribute__((counted_by(f)))
#define __counted_by_or_null(f)  __attribute__((counted_by_or_null(f)))
#define __sized_by(f)  __attribute__((sized_by(f)))

// A redeclaration or definition may rename the parameters.
void same(int n, int *__counted_by(n) p);
void same(int k, int *__counted_by(k) q) { (void)q; }
void same_out(int *len, int *__counted_by(*len) *buf);
void same_out(int *size, int *__counted_by(*size) *buf);

// Counting one-byte elements is counting bytes.
void bytes(int n, char *__counted_by(n) p);
void bytes(int n, char *__sized_by(n) p);
void void_bytes(int n, void *__counted_by(n) p);
void void_bytes(int n, void *__sized_by(n) p);

// A function without a prototype has no parameters to compare.
void no_proto(int n, int *__counted_by(n) p);
void no_proto();

// expected-note@+1{{previous declaration is here}}
void add(int n, int *p);
// expected-error@+1{{conflicting 'counted_by' attribute with the previous function declaration}}
void add(int n, int *__counted_by(n) p);

// expected-note@+1{{previous declaration is here}}
void drop(int n, int *__counted_by(n) p);
// expected-error@+1{{conflicting 'counted_by' attribute with the previous function declaration}}
void drop(int n, int *p);

// A count compares the parameter it names by position and type, qualifiers
// included.
// expected-note@+1{{previous declaration is here}}
void qualified_count(int n, int *__counted_by(n) p);
// expected-error@+1{{conflicting 'counted_by' attribute with the previous function declaration}}
void qualified_count(const int n, int *__counted_by(n) p);

// expected-note@+1{{previous declaration is here}}
void other_param(int n, int m, int *__counted_by(n) p);
// expected-error@+1{{conflicting 'counted_by' attribute with the previous function declaration}}
void other_param(int n, int m, int *__counted_by(m) p);

// expected-note@+1{{previous declaration is here}}
void kind(int n, int *__counted_by(n) p);
// expected-error@+1{{conflicting 'sized_by' attribute with the previous function declaration}}
void kind(int n, int *__sized_by(n) p);

// expected-note@+1{{previous declaration is here}}
void or_null(int n, int *__counted_by(n) p);
// expected-error@+1{{conflicting 'counted_by_or_null' attribute with the previous function declaration}}
void or_null(int n, int *__counted_by_or_null(n) p);

// The size of an incomplete element is unknown.
struct incomplete;
// expected-note@+1{{previous declaration is here}}
void incomplete_bytes(int n, struct incomplete *__counted_by(n) p);
// expected-error@+1{{conflicting 'sized_by' attribute with the previous function declaration}}
void incomplete_bytes(int n, struct incomplete *__sized_by(n) p);

// expected-note@+1{{previous declaration is here}}
void out_drop(int n, int *__counted_by(n) *buf);
// expected-error@+1{{conflicting 'counted_by' attribute with the previous function declaration}}
void out_drop(int n, int **buf);

// expected-note@+1{{previous declaration is here}}
void out_deref(int n, int *len, int *__counted_by(*len) *buf);
// expected-error@+1{{conflicting 'counted_by' attribute with the previous function declaration}}
void out_deref(int n, int *len, int *__counted_by(n) *buf);

// A K&R definition drops its counts.
// expected-note@+1{{previous declaration is here}}
void knr(int n, int *__counted_by(n) p);
// expected-error@+1{{conflicting 'counted_by' attribute with the previous function declaration}}
void knr(n, p) int n; int *__counted_by(n) p; {}

// A count naming a parameter of an enclosing function compares as that
// parameter, not by its position.
// expected-note@+1{{previous declaration is here}}
void captured(int n, int *__counted_by(n) p);
void outer(int c) {
  // expected-error@+1{{conflicting 'counted_by' attribute with the previous function declaration}}
  void captured(int n, int *__counted_by(c) p);
  void captured_twice(int n, int *__counted_by(c) p);
  void captured_twice(int k, int *__counted_by(c) q);
}

// A function declared with a typedef or '__typeof__' has its own parameters,
// but its counts name the typedef's, which compare by position as well.
typedef void fn_type(int n, int *__counted_by(n) p);
fn_type from_typedef;
void from_typedef(int n, int *__counted_by(n) p);
void from_typeof_source(int n, int *__counted_by(n) p);
__typeof__(from_typeof_source) from_typeof;
void from_typeof(int n, int *__counted_by(n) p);
void typedef_in_body(int c) {
  typedef void local_fn_type(int n, int *__counted_by(n) p);
  local_fn_type local_from_typedef;
  void local_from_typedef(int n, int *__counted_by(n) p);
}

#ifdef LATE_PARSING
// A late-parsed count that is rejected makes its declaration invalid, so
// redeclarations are not compared with it.
int global_count;
// expected-error@+1{{count expression in function declaration may only reference function parameters}}
void rejected_first(int *__counted_by(global_count) p, int n);
void rejected_first(int *__counted_by(n) p, int n);
void rejected_later(int *__counted_by(n) p, int n);
// expected-error@+1{{count expression in function declaration may only reference function parameters}}
void rejected_later(int *__counted_by(global_count) p, int n);
#endif

// Library functions, declared implicitly or in system headers, may be
// redeclared with counts. A builtin's count names a later parameter.
#ifdef LATE_PARSING
void *memcpy(void *__sized_by(n) dst, const void *__sized_by(n) src,
             __SIZE_TYPE__ n);
#endif

# 1 "system.h" 1 3
void sys_add(int n, int *p);
void sys_drop(int n, int *__counted_by(n) p);
# 132 "attr-counted-by-function-params-redecl.c" 2

void sys_add(int n, int *__counted_by(n) p);
void sys_drop(int n, int *p);

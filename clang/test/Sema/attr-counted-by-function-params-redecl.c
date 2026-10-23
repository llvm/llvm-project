// RUN: %clang_cc1 -fsyntax-only -Wno-deprecated-non-prototype -verify %s
// RUN: %clang_cc1 -fexperimental-late-parse-attributes -fsyntax-only -Wno-deprecated-non-prototype -verify %s

// Counts are part of a function's interface, so a redeclaration has to repeat
// them, also on the pointers that a parameter points to. Each count names an
// earlier parameter, so it resolves with or without late parsing.

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

// FIXME: A count naming a parameter of an enclosing function should compare as
// that parameter, not by its position.
void captured(int n, int *__counted_by(n) p);
void outer(int c) {
  void captured(int n, int *__counted_by(c) p);
  void captured_twice(int n, int *__counted_by(c) p);
  void captured_twice(int k, int *__counted_by(c) q);
}

// A function declared with a typedef or '__typeof__' gets parameters of its
// own, and the counts in its type are rebuilt to name them.
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
// Also when they take a callback's type, whose parameters are written in a
// prototype at another depth.
void (*callback_var)(int n, int *__counted_by(n) p);
__typeof__(*callback_var) from_callback;
void callback_param(void (*cb)(int n, int *__counted_by(n) p)) {
  __typeof__(*cb) from_callback;
}

// Functions declared in system headers may be redeclared with counts.

# 1 "system.h" 1 3
void sys_add(int n, int *p);
void sys_drop(int n, int *__counted_by(n) p);
# 119 "attr-counted-by-function-params-redecl.c" 2

void sys_add(int n, int *__counted_by(n) p);
void sys_drop(int n, int *p);

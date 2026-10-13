// RUN: %clang_cc1 -fexperimental-late-parse-attributes -fsyntax-only -verify %s

// A count inside a function pointer field's type is not the field's own count:
// on the callback's parameter it is completed at the end of the callback's
// parameter clause, and on its return type when the callback's type is built.
// Either way it names the callback's parameters, which the record's fields do
// not shadow.

#define __counted_by(f)  __attribute__((counted_by(f)))
#define __sized_by(f)  __attribute__((sized_by(f)))

typedef int *int_ptr;
int global_len;

struct valid {
  int len;
  void (*param)(int *__counted_by(m) p, int m);
  int *__counted_by(m) (*ret)(int m);
  // The callback's parameter, not the field of the same name.
  void *__sized_by(len) (*ret_shadows_field)(int len);
  // Each declarator names its own callback's parameters.
  void *__sized_by(l1) (*f1)(unsigned l1), *__sized_by(l2) (*f2)(unsigned l2);
  // A count in the declaration specifiers, which the declarators share, is not
  // parsed with a callback's parameters.
  // expected-error@+1 2{{'counted_by' attribute on nested pointer type is not allowed}}
  int_ptr __counted_by(n) (*g1)(int n), (*g2)(int n);
};

struct names_field {
  int len;
  // expected-error@+1{{use of undeclared identifier 'len'}}
  void (*param)(int *__counted_by(len) p);
  // expected-error@+1{{use of undeclared identifier 'len'}}
  int *__counted_by(len) (*ret)(int m);
};

struct names_global {
  // expected-error@+1{{count expression in function declaration may only reference function parameters}}
  void (*param)(int *__counted_by(global_len) p);
  // expected-error@+1{{argument of 'counted_by' attribute cannot refer to declaration from a different scope}}
  int *__counted_by(global_len) (*ret)(void);
};

// A count shared through the declaration specifiers is the field's own count
// for 'a', naming the field, and is rejected in the return type of 'b'.
struct shared_own_and_return {
  int n;
  // expected-error@+1{{'counted_by' attribute on nested pointer type is not allowed}}
  int_ptr __counted_by(n) a, (*b)(void);
};

struct nested_callbacks {
  void (*outer)(int n, void (*inner)(int *__counted_by(n) p));
  int *__counted_by(k) (*(*make)(int j))(int k);
};

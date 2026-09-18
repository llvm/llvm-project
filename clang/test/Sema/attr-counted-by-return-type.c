// RUN: %clang_cc1 -fblocks -fsyntax-only -verify %s

// counted_by-family attributes aren't supported on function return types,
// including the return type of the function a function-pointer or
// block-pointer member points to. They're rejected with a dedicated diagnostic.
// They used to be reported as a nested pointer, because the function chunk
// counted as a level of pointer nesting.

#define __counted_by(f) __attribute__((counted_by(f)))
#define __sized_by(f) __attribute__((sized_by(f)))
#define __counted_by_or_null(f) __attribute__((counted_by_or_null(f)))
#define __sized_by_or_null(f) __attribute__((sized_by_or_null(f)))

extern int g;

// expected-error@+1{{'counted_by' attribute cannot be applied to a function return type}}
int *__counted_by(g) fn(void);
// expected-error@+1{{'sized_by' attribute cannot be applied to a function return type}}
char *__sized_by(4) fn_const(void);

struct fn_ptr_return {
  int n;
  // expected-error@+1{{'counted_by' attribute cannot be applied to a function return type}}
  int *__counted_by(n) (*fp)(void);
};

struct block_ptr_return {
  int n;
  // expected-error@+1{{'counted_by_or_null' attribute cannot be applied to a function return type}}
  int *__counted_by_or_null(n) (^bp)(void);
};

struct fn_ptr_array_return {
  int n;
  // expected-error@+1{{'sized_by_or_null' attribute cannot be applied to a function return type}}
  char *__sized_by_or_null(n) (*fpa[2])(void);
};

// A counted pointer nested anywhere in a return type is rejected as being on
// the return type. Its nesting is measured relative to the return type.
struct fn_ptr_return_nested {
  int n;
  // expected-error@+1{{'counted_by' attribute cannot be applied to a function return type}}
  int *__counted_by(n) *(*fp)(void);
};

// Nesting stops at the function: the pointer to the function and the array
// holding it don't count toward the return type's nesting.
struct fn_ptr_array_nested_return {
  int n;
  // expected-error@+1{{'counted_by' attribute cannot be applied to a function return type}}
  int *__counted_by(n) *(*fpa[2])(void);
};

// RUN: %clang_cc1 -fblocks -fsyntax-only -verify=expected,eager %s
// RUN: %clang_cc1 -fblocks -fexperimental-late-parse-attributes -fsyntax-only -verify=expected,late %s

// counted_by-family attributes aren't supported on function return types,
// including the return type of the function a function-pointer or
// block-pointer member points to. Both paths reject them with a dedicated
// diagnostic. They used to be reported as a nested pointer without late
// parsing, and silently accepted with it.

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

// A pointer to a counted pointer is still a nested pointer, even in a return
// type.
struct fn_ptr_return_nested {
  int n;
  // expected-error@+1{{'counted_by' attribute on nested pointer type is not allowed}}
  int *__counted_by(n) *(*fp)(void);
};

// Without late parsing, the count is looked up when the attribute is parsed.
// With it, the rejected attribute's argument is never parsed.
struct fn_ptr_return_count_after {
  // late-error@+2{{'counted_by' attribute cannot be applied to a function return type}}
  // eager-error@+1{{use of undeclared identifier 'n'}}
  int *__counted_by(n) (*fp)(void);
  int n;
};

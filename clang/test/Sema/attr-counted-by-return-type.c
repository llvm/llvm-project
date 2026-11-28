// RUN: %clang_cc1 -fblocks -fsyntax-only -verify %s

// counted_by-family attributes aren't supported yet on function return types,
// including the return type of the function a function-pointer or
// block-pointer member points to. They're rejected with a dedicated diagnostic.

#define __counted_by(f) __attribute__((counted_by(f)))
#define __sized_by(f) __attribute__((sized_by(f)))
#define __counted_by_or_null(f) __attribute__((counted_by_or_null(f)))
#define __sized_by_or_null(f) __attribute__((sized_by_or_null(f)))

extern int g;

// Outside a struct, the attribute isn't on a field, so it's diagnosed as such
// before its position is considered.
// expected-error@+1{{'counted_by' attribute only applies to non-static data members}}
int *__counted_by(g) fn(void);
// expected-error@+1{{'sized_by' attribute only applies to non-static data members}}
char *__sized_by(4) fn_const(void);

struct fn_ptr_return {
  int n;
  // expected-error@+1{{'counted_by' attribute on a function return type is not yet supported}}
  int *__counted_by(n) (*fp)(void);
};

struct block_ptr_return {
  int n;
  // expected-error@+1{{'counted_by_or_null' attribute on a function return type is not yet supported}}
  int *__counted_by_or_null(n) (^bp)(void);
};

struct fn_ptr_array_return {
  int n;
  // expected-error@+1{{'sized_by_or_null' attribute on a function return type is not yet supported}}
  char *__sized_by_or_null(n) (*fpa[2])(void);
};

// A counted pointer nested anywhere in a return type is rejected as being on
// the return type. Its nesting is measured relative to the return type.
struct fn_ptr_return_nested {
  int n;
  // expected-error@+1{{'counted_by' attribute on a function return type is not yet supported}}
  int *__counted_by(n) *(*fp)(void);
};

// Nesting stops at the function: the pointer to the function and the array
// holding it don't count toward the return type's nesting.
struct fn_ptr_array_nested_return {
  int n;
  // expected-error@+1{{'counted_by' attribute on a function return type is not yet supported}}
  int *__counted_by(n) *(*fpa[2])(void);
};

// A function pointer whose return type is another function pointer.
struct fn_ptr_returning_fn_ptr {
  int n;
  // expected-error@+1{{'counted_by' attribute on a function return type is not yet supported}}
  int *__counted_by(n) (*(*fpfp)(void))(void);
};

// The GNU spelling reaches the array that a function returns a pointer to when
// it's written after the outermost '(' of the declarator. That puts it on the
// parentheses chunk, whose type is the array.
struct fn_ptr_returning_ptr_to_array {
  int n;
  // expected-error@+1{{'counted_by' attribute on a function return type is not yet supported}}
  int (__counted_by(n) *(*(*fpfp_arr)(void))(void))[];
  // The position is checked before whether the array can be counted.
  // expected-error@+1{{'counted_by' attribute on a function return type is not yet supported}}
  int (__counted_by(n) *(*(*fpfp_arr4)(void))(void))[4];
  // A pointer to an array of function pointers that return a pointer to the
  // array.
  // expected-error@+1{{'counted_by' attribute on a function return type is not yet supported}}
  int (__counted_by(n) *(*(*pafp)[4])(void))[];
  // After an inner '(' it's on the function type that a pointer points to,
  // which isn't a pointer.
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int (*(__counted_by(n) *(*fp_inner)(void))(void))[];
};

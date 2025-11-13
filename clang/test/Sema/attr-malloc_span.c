// RUN: %clang_cc1 -verify -fsyntax-only %s
// RUN: %clang_cc1 -emit-llvm -o %t %s

typedef __SIZE_TYPE__ size_t;

typedef struct {
  void *ptr;
  size_t n;
} span;
span  returns_span  (void) __attribute((malloc_span)); // no-warning

// Try out a different field ordering.
typedef struct {
  size_t n;
  void *ptr;
} span2;
span2  returns_span2  (void) __attribute((malloc_span)); // no-warning

typedef struct {
  void *ptr;
  void *ptr2;
} span3;
span3  returns_span3  (void) __attribute((malloc_span)); // no-warning
// expected-warning@+2 {{attribute only applies to functions that return span-like structures}}
// expected-note@+1 {{span-like type must be a struct}}
int *returns_int_ptr  (void) __attribute((malloc_span));

typedef struct {
  void *ptr;
  size_t n;
  size_t n2;
} too_long_span;
// expected-warning@+2 {{attribute only applies to functions that return span-like structures}}
// expected-note@+1 {{span-like type must have 2 fields}}
too_long_span  returns_too_long_span  (void) __attribute((malloc_span));

// Function pointers are not allowed.
typedef struct {
  int (*func_ptr)(void);
  size_t n;
} func_span;
// expected-warning@+2 {{attribute only applies to functions that return span-like structures}}
// expected-note@+1 {{span-like type must have a pointer and an integer field or two pointer fields}}
func_span  returns_func_span  (void) __attribute((malloc_span));

// Integer should not be an enum.
enum some_enum { some_value, other_value };
typedef struct {
  void *ptr;
  enum some_enum field;
} enum_span;
// expected-warning@+2 {{attribute only applies to functions that return span-like structures}}
// expected-note@+1 {{the integer field must be an actual integer}}
enum_span  returns_enum_span  (void) __attribute((malloc_span));

// Bit integers are also not supported.
typedef struct {
  void *ptr;
  _BitInt(16) n;
} bit_span;
// expected-warning@+2 {{attribute only applies to functions that return span-like structures}}
// expected-note@+1 {{the integer field must be an actual integer}}
bit_span  returns_bit_span  (void) __attribute((malloc_span));

// Integer must be at least as big as int.
typedef struct {
  void *ptr;
  short n;
} short_span;
// expected-warning@+2 {{attribute only applies to functions that return span-like structures}}
// expected-note@+1 {{the integer field must be an actual integer}}
short_span  returns_short_span  (void) __attribute((malloc_span));

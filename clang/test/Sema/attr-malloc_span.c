// RUN: %clang_cc1 -verify -fsyntax-only %s

typedef __SIZE_TYPE__ size_t;

typedef struct {
  void *ptr;
  size_t n;
} span;
span  returns_span  (void) __attribute((malloc_span)); // no-warning

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

typedef struct {
  void *ptr;
  int n;
} span4;
span4  returns_span4  (void) __attribute((malloc_span)); // no-warning

typedef struct incomplete_span incomplete_span;
// expected-warning@+2 {{attribute only applies to functions that return span-like structures}}
// expected-note@+1 {{returned type is incomplete}}
incomplete_span returns_incomplete_span (void) __attribute((malloc_span));

// expected-warning@+2 {{attribute only applies to functions that return span-like structures}}
// expected-note@+1 {{returned type is not a struct type}}
int *returns_int_ptr  (void) __attribute((malloc_span));

typedef struct {
  void *ptr;
  size_t n;
  size_t n2;
} too_long_span;
// expected-warning@+2 {{attribute only applies to functions that return span-like structures}}
// expected-note@+1 {{returned struct has 3 fields, expected 2}}
too_long_span  returns_too_long_span  (void) __attribute((malloc_span));

// Function pointers are not allowed.
typedef struct {
  int (*func_ptr)(void);
  size_t n;
} func_span;
// expected-warning@+2 {{attribute only applies to functions that return span-like structures}}
// expected-note@+1 {{returned struct fields are not a supported combination}}
func_span  returns_func_span  (void) __attribute((malloc_span));

// Integer should not be an enum.
enum some_enum { some_value, other_value };
typedef struct {
  void *ptr;
  enum some_enum field;
} enum_span;
// expected-warning@+2 {{attribute only applies to functions that return span-like structures}}
// expected-note@+1 {{2nd field is expected to be an integer}}
enum_span  returns_enum_span  (void) __attribute((malloc_span));

// Bit integers are also not supported.
typedef struct {
  void *ptr;
  _BitInt(16) n;
} bit_span;
// expected-warning@+2 {{attribute only applies to functions that return span-like structures}}
// expected-note@+1 {{2nd field is expected to be an integer}}
bit_span  returns_bit_span  (void) __attribute((malloc_span));

// Integer must be at least as big as int.
typedef struct {
  void *ptr;
  short n;
} short_span;
// expected-warning@+2 {{attribute only applies to functions that return span-like structures}}
// expected-note@+1 {{2nd field of span-like type is not a wide enough integer (minimum width: 32)}}
short_span  returns_short_span  (void) __attribute((malloc_span));

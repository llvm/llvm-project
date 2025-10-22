// XFAIL: *
// FIXME: https://github.com/llvm/llvm-project/issues/166454

// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fexperimental-late-parse-attributes -fsyntax-only -verify %s

#define __counted_by(N) __attribute__((counted_by(N)))
#define __counted_by_or_null(N) __attribute__((counted_by_or_null(N)))
#define __sized_by(N) __attribute__((sized_by(N)))
#define __sized_by_or_null(N) __attribute__((sized_by_or_null(N)))

//==============================================================================
// Test bounds safety attributes on function pointer parameters
//==============================================================================

struct counted_by_function_pointer_param {
  // expected-error@+1{{'counted_by' attribute cannot be applied to a parameter in a function pointer type}}
  int (*callback)(int *__counted_by(len));
  int len;
};

struct counted_by_or_null_function_pointer_param {
  // expected-error@+1{{'counted_by_or_null' attribute cannot be applied to a parameter in a function pointer type}}
  int (*callback)(int *__counted_by_or_null(len));
  int len;
};

struct sized_by_function_pointer_param {
  // expected-error@+1{{'sized_by' attribute cannot be applied to a parameter in a function pointer type}}
  int (*callback)(char *__sized_by(len));
  int len;
};

struct sized_by_or_null_function_pointer_param {
  // expected-error@+1{{'sized_by_or_null' attribute cannot be applied to a parameter in a function pointer type}}
  int (*callback)(char *__sized_by_or_null(len));
  int len;
};

//==============================================================================
// Test multiple parameters with bounds safety attributes
//==============================================================================

struct multiple_params_with_bounds_safety {
  // expected-error@+1{{'counted_by' attribute cannot be applied to a parameter in a function pointer type}}
  int (*multi_callback)(int *__counted_by(len1), char *data, int len1);
  int len1;
};

struct mixed_bounds_safety_params {
  // expected-error@+2{{'counted_by' attribute cannot be applied to a parameter in a function pointer type}}
  // expected-error@+1{{'sized_by_or_null' attribute cannot be applied to a parameter in a function pointer type}}
  int (*mixed_callback)(int *__counted_by(count), char *__sized_by_or_null(size), int count, int size);
  int count;
  int size;
};

//==============================================================================
// Test cases that do not require late parsing (count field defined before use)
//==============================================================================

struct counted_by_no_late_parse {
  int len;
  // expected-error@+1{{'counted_by' attribute cannot be applied to a parameter in a function pointer type}}
  int (*callback)(int *__counted_by(len));
};

struct counted_by_or_null_no_late_parse {
  int len;
  // expected-error@+1{{'counted_by_or_null' attribute cannot be applied to a parameter in a function pointer type}}
  int (*callback)(int *__counted_by_or_null(len));
};

struct sized_by_no_late_parse {
  int len;
  // expected-error@+1{{'sized_by' attribute cannot be applied to a parameter in a function pointer type}}
  int (*callback)(char *__sized_by(len));
};

struct sized_by_or_null_no_late_parse {
  int len;
  // expected-error@+1{{'sized_by_or_null' attribute cannot be applied to a parameter in a function pointer type}}
  int (*callback)(char *__sized_by_or_null(len));
};

//==============================================================================
// Test nested function pointer types
//==============================================================================

struct nested_function_pointer_with_bounds_safety {
  // expected-error@+1{{'counted_by' attribute cannot be applied to a parameter in a function pointer type}}
  int (*outer_callback)(int (*inner)(int *__counted_by(len)), int len);
  int len;
};

//==============================================================================
// Test struct members with anonymous structs/unions (no late parsing needed)
//==============================================================================

struct with_anonymous_struct_no_late_parse {
  int len;
  // expected-error@+1{{'counted_by' attribute cannot be applied to a parameter in a function pointer type}}
  int (*callback)(int *__counted_by(len));
};

struct with_anonymous_union_no_late_parse {
  union {
    int len;
    float f_len;
  };
  // expected-error@+1{{'counted_by_or_null' attribute cannot be applied to a parameter in a function pointer type}}
  int (*callback)(int *__counted_by_or_null(len));
};

//==============================================================================
// Test with different parameter positions
//==============================================================================

struct first_param_bounds_safety_no_late_parse {
  int count;
  // expected-error@+1{{'counted_by' attribute cannot be applied to a parameter in a function pointer type}}
  int (*callback)(int *__counted_by(count), void *data, int extra);
};

struct middle_param_bounds_safety_no_late_parse {
  int size;
  // expected-error@+1{{'sized_by' attribute cannot be applied to a parameter in a function pointer type}}
  int (*callback)(void *prefix, char *__sized_by(size), int suffix);
};

struct last_param_bounds_safety_no_late_parse {
  int len;
  // expected-error@+1{{'counted_by_or_null' attribute cannot be applied to a parameter in a function pointer type}}
  int (*callback)(int a, float b, int *__counted_by_or_null(len));
};

//==============================================================================
// Test with const and volatile qualifiers
//==============================================================================

struct const_param_bounds_safety_no_late_parse {
  int count;
  // expected-error@+1{{'counted_by' attribute cannot be applied to a parameter in a function pointer type}}
  int (*callback)(const int *__counted_by(count));
};

struct volatile_param_bounds_safety_no_late_parse {
  int size;
  // expected-error@+1{{'sized_by_or_null' attribute cannot be applied to a parameter in a function pointer type}}
  int (*callback)(volatile char *__sized_by_or_null(size));
};

struct const_volatile_param_bounds_safety_no_late_parse {
  int len;
  // expected-error@+1{{'counted_by_or_null' attribute cannot be applied to a parameter in a function pointer type}}
  int (*callback)(const volatile int *__counted_by_or_null(len));
};

//==============================================================================
// Test with multiple function pointers in same struct
//==============================================================================

struct multiple_function_pointers_no_late_parse {
  int len1, len2, size1, size2;
  // expected-error@+1{{'counted_by' attribute cannot be applied to a parameter in a function pointer type}}
  int (*callback1)(int *__counted_by(len1));
  // expected-error@+1{{'counted_by_or_null' attribute cannot be applied to a parameter in a function pointer type}}
  int (*callback2)(int *__counted_by_or_null(len2));
  // expected-error@+1{{'sized_by' attribute cannot be applied to a parameter in a function pointer type}}
  void (*callback3)(char *__sized_by(size1));
  // expected-error@+1{{'sized_by_or_null' attribute cannot be applied to a parameter in a function pointer type}}
  void (*callback4)(char *__sized_by_or_null(size2));
};

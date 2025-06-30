// RUN: %clang_cc1 -Wall -Wno-unused -Wno-uninitialized -verify %s

#define CFI_UNCHECKED_CALLEE __attribute__((cfi_unchecked_callee))

void unchecked(void) CFI_UNCHECKED_CALLEE {}
void checked(void) {}

void (*checked_ptr)(void) = unchecked;  // expected-warning{{implicit conversion from 'void (void) __attribute__((cfi_unchecked_callee))' to 'void (*)(void)' discards 'cfi_unchecked_callee' attribute}}
void (CFI_UNCHECKED_CALLEE *unchecked_ptr)(void) = unchecked;
void (CFI_UNCHECKED_CALLEE *from_normal)(void) = checked;
void (CFI_UNCHECKED_CALLEE *c_no_function_decay)(void) = &unchecked;

typedef void (CFI_UNCHECKED_CALLEE unchecked_func_t)(void);
typedef void (checked_func_t)(void);
typedef void (CFI_UNCHECKED_CALLEE *cfi_unchecked_func_ptr_t)(void);
typedef void (*checked_func_ptr_t)(void);
checked_func_t *cfi_func = unchecked;  // expected-warning{{implicit conversion from 'void (void) __attribute__((cfi_unchecked_callee))' to 'void (*)(void)' discards 'cfi_unchecked_callee' attribute}}
unchecked_func_t *unchecked_func = unchecked;

void CFI_UNCHECKED_CALLEE after_ret_type(void);
CFI_UNCHECKED_CALLEE void before_ret_type(void);
void (* CFI_UNCHECKED_CALLEE after_name)(void);

void UsageOnImproperTypes() {
  int CFI_UNCHECKED_CALLEE i;  // expected-warning{{'cfi_unchecked_callee' only applies to function types; type here is 'int'}}

  /// The attribute must be used only on functions with prototypes. The omission of `void` means it is not prototyped.
  void (CFI_UNCHECKED_CALLEE *noproto)(void);  // expecteed-warning{{'cfi_unchecked_callee' attribute only applies to non-K&R-style functions}}
}

/// Explicit casts suppress the warning.
void CheckCasts() {
  void (*should_warn)(void) = unchecked;  // expected-warning{{implicit conversion from 'void (void) __attribute__((cfi_unchecked_callee))' to 'void (*)(void)' discards 'cfi_unchecked_callee' attribute}}

  void (*no_warn_c_style_cast)(void) = (void (*)(void))unchecked;

  struct B {} CFI_UNCHECKED_CALLEE b;  // expected-warning{{'cfi_unchecked_callee' attribute only applies to functions and methods}}
  struct CFI_UNCHECKED_CALLEE C {} c;  // expected-warning{{'cfi_unchecked_callee' attribute only applies to functions and methods}}
  CFI_UNCHECKED_CALLEE struct D {} d;  // expected-warning{{'cfi_unchecked_callee' only applies to function types; type here is 'struct D'}}

  void *ptr2 = (void *)unchecked;
}

int checked_arg_func(checked_func_t *checked_func);

void CheckDifferentConstructions() {
  void (CFI_UNCHECKED_CALLEE *arr[10])(void);
  void (*cfi_elem)(void) = arr[1];  // expected-warning{{implicit conversion from 'void (*)(void) __attribute__((cfi_unchecked_callee))' to 'void (*)(void)' discards 'cfi_unchecked_callee' attribute}}
  void (CFI_UNCHECKED_CALLEE *cfi_unchecked_elem)(void) = arr[1];

  int invoke = checked_arg_func(unchecked);  // expected-warning{{implicit conversion from 'void (void) __attribute__((cfi_unchecked_callee))' to 'void (*)(void)' discards 'cfi_unchecked_callee' attribute}}
}

checked_func_t *returning_checked_func() {
  return unchecked;  // expected-warning{{implicit conversion from 'void (void) __attribute__((cfi_unchecked_callee))' to 'void (*)(void)' discards 'cfi_unchecked_callee' attribute}}
}

void no_args(void) __attribute__((cfi_unchecked_callee(10)));  // expected-error{{'cfi_unchecked_callee' attribute takes no arguments}}

void Comparisons() {
  /// Let's be able to compare checked and unchecked pointers without warnings.
  unchecked == checked_ptr;
  checked_ptr == unchecked;
  unchecked == unchecked_ptr;
  unchecked != checked_ptr;
  checked_ptr != unchecked;
  unchecked != unchecked_ptr;

  (void (*)(void))unchecked == checked_ptr;
  checked_ptr == (void (*)(void))unchecked;
}

// RUN: %clang_cc1 -fsyntax-only %s -verify

[[noreturn]] extern void noret();
[[noreturn]] extern void noret2();
extern void ordinary();

typedef void (*func_type)(void);

// Constant initialization.

void (* const const_fptr)() = noret;
[[noreturn]] void test_global_const() {
  const_fptr();
}

const func_type const_fptr_cast = (func_type)noret2;
[[noreturn]] void test_global_cast() {
  const_fptr_cast();
}

void (* const const_fptr_list)() = {noret};
[[noreturn]] void test_global_list() {
  const_fptr_list();
}

const func_type const_fptr_fcast = func_type(noret2);
[[noreturn]] void test_global_fcast() {
  const_fptr_fcast();
}

[[noreturn]] void test_local_const() {
  void (* const fptr)() = noret;
  fptr();
}

// Global variable assignment.
void (*global_fptr)() = noret;

[[noreturn]] void test_global_noassign() {
  global_fptr();
} // expected-warning {{function declared 'noreturn' should not return}}

[[noreturn]] void test_global_assign() {
  global_fptr = noret;
  global_fptr();
} // expected-warning {{function declared 'noreturn' should not return}}

// Local variable assignment.

[[noreturn]] void test_init() {
  func_type func_ptr = noret;
  func_ptr();
}

[[noreturn]] void test_assign() {
  void (*func_ptr)(void);
  func_ptr = noret;
  func_ptr();
}

[[noreturn]] void test_override() {
  func_type func_ptr;
  func_ptr = ordinary;
  func_ptr = noret;
  func_ptr();
}

[[noreturn]] void test_if_all(int x) {
  func_type func_ptr;
  if (x > 0)
    func_ptr = noret;
  else
    func_ptr = noret2;
  func_ptr();
}

[[noreturn]] void test_if_mix(int x) {
  func_type func_ptr;
  if (x > 0)
    func_ptr = noret;
  else
    func_ptr = ordinary;
  func_ptr();
} // expected-warning {{function declared 'noreturn' should not return}}

[[noreturn]] void test_if_opt(int x) {
  func_type func_ptr = noret;
  if (x > 0)
    func_ptr = ordinary;
  func_ptr();
} // expected-warning {{function declared 'noreturn' should not return}}

[[noreturn]] void test_if_opt2(int x) {
  func_type func_ptr = ordinary;
  if (x > 0)
    func_ptr = noret;
  func_ptr();
} // expected-warning {{function declared 'noreturn' should not return}}

[[noreturn]] void test_if_nest_all(int x, int y) {
  func_type func_ptr;
  if (x > 0) {
    if (y > 0)
      func_ptr = noret;
    else
      func_ptr = noret2;
  } else {
    if (y < 0)
      func_ptr = noret2;
    else
      func_ptr = noret;
  }
  func_ptr();
}

[[noreturn]] void test_if_nest_mix(int x, int y) {
  func_type func_ptr;
  if (x > 0) {
    if (y > 0)
      func_ptr = noret;
    else
      func_ptr = noret2;
  } else {
    if (y < 0)
      func_ptr = ordinary;
    else
      func_ptr = noret;
  }
  func_ptr();
} // expected-warning {{function declared 'noreturn' should not return}}

[[noreturn]] void test_switch_all(int x) {
  func_type func_ptr;
  switch(x) {
  case 1:
    func_ptr = noret;
    break;
  default:
    func_ptr = noret2;
    break;
  }
  func_ptr();
}

[[noreturn]] void test_switch_mix(int x) {
  func_type func_ptr;
  switch(x) {
  case 1:
    func_ptr = ordinary;
    break;
  default:
    func_ptr = noret;
    break;
  }
  func_ptr();
} // expected-warning {{function declared 'noreturn' should not return}}

[[noreturn]] void test_switch_fall(int x) {
  func_type func_ptr;
  switch(x) {
  case 1:
    func_ptr = ordinary;
  default:
    func_ptr = noret;
    break;
  }
  func_ptr();
}

[[noreturn]] void test_switch_all_nest(int x, int y) {
  func_type func_ptr;
  switch(x) {
  case 1:
    func_ptr = noret;
    break;
  default:
    if (y > 0)
      func_ptr = noret2;
    else
      func_ptr = noret;
    break;
  }
  func_ptr();
}

[[noreturn]] void test_switch_mix_nest(int x, int y) {
  func_type func_ptr;
  switch(x) {
  case 1:
    func_ptr = noret;
    break;
  default:
    if (y > 0)
      func_ptr = noret2;
    else
      func_ptr = ordinary;
    break;
  }
  func_ptr();
} // expected-warning {{function declared 'noreturn' should not return}}

// Function parameters.

[[noreturn]] void test_param(void (*func_ptr)() = noret) {
  func_ptr();
} // expected-warning {{function declared 'noreturn' should not return}}

[[noreturn]] void test_const_param(void (* const func_ptr)() = noret) {
  func_ptr();
} // expected-warning {{function declared 'noreturn' should not return}}

// Escaped value.

extern void abc_01(func_type &);
extern void abc_02(func_type *);

[[noreturn]] void test_escape_ref() {
  func_type func_ptr = noret;
  abc_01(func_ptr);
  func_ptr();
} // expected-warning {{function declared 'noreturn' should not return}}

[[noreturn]] void test_escape_addr() {
  func_type func_ptr = noret;
  abc_02(&func_ptr);
  func_ptr();
} // expected-warning {{function declared 'noreturn' should not return}}

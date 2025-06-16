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
}

[[noreturn]] void test_global_override() {
  global_fptr = ordinary;
  global_fptr = noret;
  global_fptr();
}

[[noreturn]] void test_global_switch_01(int x) {
  switch(x) {
  case 1:
    global_fptr = noret;
	break;
  default:
    global_fptr = noret2;
	break;
  }
  global_fptr();
}

[[noreturn]] void test_global_switch_02(int x) {
  switch(x) {
  case 1:
    global_fptr = ordinary;
	break;
  default:
    global_fptr = noret;
	break;
  }
  global_fptr();
}

// Local variable assignment.

[[noreturn]] void test_local_init() {
  func_type func_ptr = noret;
  func_ptr();
}

[[noreturn]] void test_local_assign() {
  void (*func_ptr)(void);
  func_ptr = noret;
  func_ptr();
}

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


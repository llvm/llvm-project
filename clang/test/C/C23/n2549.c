// RUN: %clang_cc1 -verify -std=c23 -Wno-gnu-line-marker %s
// RUN: %clang_cc1 -verify=pedantic -std=c17 -pedantic -Wno-gnu-line-marker %s
// RUN: %clang_cc1 -verify=compat -std=c23 -Wpre-c23-compat -Wno-gnu-line-marker %s

// expected-no-diagnostics

/* WG14 N2549: Clang 9
 * Binary literals
 */

int i = 0b01; /* pedantic-warning {{binary integer literals are a C23 extension}}
                 compat-warning {{binary integer literals are incompatible with C standards before C23}}
               */

int other_func(int i, ...);
#16 "system_header.h" 3
#define SYS_HEADER_MACRO_1 0b1010
#define SYS_HEADER_MACRO_2 1010
#define SYS_HEADER_MACRO_FN1(...) other_func(0b1010, __VA_ARGS__)
#define SYS_HEADER_MACRO_FN2(...) other_func(1010, __VA_ARGS__)

#22 "n2549.c" 1

#define USER_HEADER_MACRO_1 0b1010
#define USER_HEADER_MACRO_2 1010
#define USER_HEADER_MACRO_FN1(...) other_func(0b1010, __VA_ARGS__)
#define USER_HEADER_MACRO_FN2(...) other_func(1010, __VA_ARGS__)

#define USER_OBJECT_MACRO SYS_HEADER_MACRO_1
#define USER_FUNCTION_MACRO() SYS_HEADER_MACRO_1

void test_binary_macro_behavior(void) {
  int i = SYS_HEADER_MACRO_1;
  int j = SYS_HEADER_MACRO_2;
  int k = SYS_HEADER_MACRO_FN1(12);

  int l = SYS_HEADER_MACRO_FN2(0b1010); /* pedantic-warning {{binary integer literals are a C23 extension}}
                 compat-warning {{binary integer literals are incompatible with C standards before C23}}
               */

  int m = USER_OBJECT_MACRO;
  int n = USER_FUNCTION_MACRO();

  int a = USER_HEADER_MACRO_1; /* pedantic-warning {{binary integer literals are a C23 extension}}
                 compat-warning {{binary integer literals are incompatible with C standards before C23}}
               */
  int b = USER_HEADER_MACRO_FN1(12); /* pedantic-warning {{binary integer literals are a C23 extension}}
                 compat-warning {{binary integer literals are incompatible with C standards before C23}}
               */
  int c = USER_HEADER_MACRO_FN2(0b1010); /* pedantic-warning {{binary integer literals are a C23 extension}}
                 compat-warning {{binary integer literals are incompatible with C standards before C23}}
               */
}

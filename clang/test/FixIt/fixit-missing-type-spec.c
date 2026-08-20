// RUN: %clang_cc1 -std=c90 -pedantic -Wno-comment -Wno-deprecated-non-prototype -Wimplicit-int -fsyntax-only -verify=c90 -x c %s
// RUN: cp %s %t
// RUN: not %clang_cc1 -std=c90 -pedantic -Wno-comment -Wno-deprecated-non-prototype -Wimplicit-int -Werror -fixit %t
// RUN: %clang_cc1 -std=c90 -pedantic -Wno-comment -Wno-deprecated-non-prototype -Wimplicit-int -fsyntax-only -verify -x c %t
// RUN: cat %t | FileCheck %s
//
// RUN: %clang_cc1 -std=c99 -pedantic -Wno-deprecated-non-prototype -fsyntax-only -verify=c99 -x c %s
// RUN: cp %s %t
// RUN: not %clang_cc1 -std=c99 -pedantic -Wno-deprecated-non-prototype -fixit %t
// RUN: %clang_cc1 -std=c99 -pedantic -Wno-deprecated-non-prototype -fsyntax-only -verify -x c %t
// RUN: cat %t | FileCheck %s
//
// RUN: %clang_cc1 -std=c23 -pedantic -fsyntax-only -verify=c23 -x c %s

// expected-no-diagnostics

// CHECK: int imp0[4],imp1,imp2=5;
imp0[4],imp1,imp2=5;
// c90-warning@-1 {{type specifier missing, defaults to 'int'}}
// c99-error@-2 {{type specifier missing, defaults to 'int'}}
// c23-error@-3 {{a type specifier is required for all declarations}}
// c23-error@-4 {{expected ';' after top level declarator}}

// CHECK: int const imp3;
const imp3;
// c90-warning@-1 {{type specifier missing, defaults to 'int'}}
// c99-error@-2 {{type specifier missing, defaults to 'int'}}
// c23-error@-3 {{a type specifier is required for all declarations}}

// CHECK: int static imp4;
static imp4;
// c90-warning@-1 {{type specifier missing, defaults to 'int'}}
// c99-error@-2 {{type specifier missing, defaults to 'int'}}
// c23-error@-3 {{a type specifier is required for all declarations}}

// CHECK: int static const imp5;
static const imp5;
// c90-warning@-1 {{type specifier missing, defaults to 'int'}}
// c99-error@-2 {{type specifier missing, defaults to 'int'}}
// c23-error@-3 {{a type specifier is required for all declarations}}

// CHECK: int volatile __attribute__ ((aligned (16))) imp6;
volatile __attribute__ ((aligned (16))) imp6;
// c90-warning@-1 {{type specifier missing, defaults to 'int'}}
// c99-error@-2 {{type specifier missing, defaults to 'int'}}
// c23-error@-3 {{a type specifier is required for all declarations}}

// CHECK-LABEL: int f2(void)
f2(void)
// c90-warning@-1 {{type specifier missing, defaults to 'int'}}
// c99-error@-2 {{type specifier missing, defaults to 'int'}}
// c23-error@-3 {{a type specifier is required for all declarations}}
{
  // CHECK: int register __attribute__ ((uninitialized)) i;
  register __attribute__ ((uninitialized)) i;
  // c90-warning@-1 {{type specifier missing, defaults to 'int'}}
  // c99-error@-2 {{type specifier missing, defaults to 'int'}}
  // c23-error@-3 {{a type specifier is required for all declarations}}

  return 0;
}

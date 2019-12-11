// RUN: %clang_cc1 -fsyntax-only -Wdocumentation -Wdocumentation-pedantic -fcomment-block-commands=foobar -verify %s
// RUN  %clang_cc1 -std=c18 -fsyntax-only -Wdocumentation -Wdocumentation-pedantic -fcomment-block-commands=foobar -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck -DATTRIBUTE="__attribute__((deprecated))" %s
// RUN: %clang_cc1 -std=c2x -DC2x -fsyntax-only -Wdocumentation -Wdocumentation-pedantic -fcomment-block-commands=foobar -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck --check-prefixes=CHECK,CHECK2x -DATTRIBUTE="[[deprecated]]" %s

// expected-warning@+1 {{declaration is marked with '\deprecated' command but does not have a deprecation attribute}} expected-note@+2 {{add a deprecation attribute to the declaration to silence this warning}}
/// \deprecated
void test_deprecated_1();

// expected-warning@+1 {{declaration is marked with '\deprecated' command but does not have a deprecation attribute}} expected-note@+2 {{add a deprecation attribute to the declaration to silence this warning}}
/// \deprecated
void test_deprecated_2(int a);

#define MY_ATTR_DEPRECATED __attribute__((deprecated))

// expected-warning@+1 {{declaration is marked with '\deprecated' command but does not have a deprecation attribute}} expected-note@+2 {{add a deprecation attribute to the declaration to silence this warning}}
/// \deprecated
void test_deprecated_3(int a);

#ifdef C2x
#define ATTRIBUTE_DEPRECATED [[deprecated]]

// expected-warning@+1 {{declaration is marked with '\deprecated' command but does not have a deprecation attribute}} expected-note@+2 {{add a deprecation attribute to the declaration to silence this warning}}
/// \deprecated
void test_deprecated_4(int a);
#endif

// CHECK: fix-it:"{{.*}}":{7:1-7:1}:"[[ATTRIBUTE]] "
// CHECK: fix-it:"{{.*}}":{11:1-11:1}:"[[ATTRIBUTE]] "
// CHECK: fix-it:"{{.*}}":{17:1-17:1}:"MY_ATTR_DEPRECATED "
// CHECK2x: fix-it:"{{.*}}":{24:1-24:1}:"ATTRIBUTE_DEPRECATED "

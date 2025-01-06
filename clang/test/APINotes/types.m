// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fapinotes-modules -Wno-private-module -fdisable-module-hash -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -verify
// RUN: %clang_cc1 -ast-print %t/ModulesCache/SimpleKit.pcm | FileCheck %s

#import <SomeKit/SomeKit.h>
#import <SimpleKit/SimpleKit.h>

// CHECK: struct __attribute__((swift_name("SuccessfullyRenamedA"))) RenamedAgainInAPINotesA {
// CHECK: struct __attribute__((swift_name("SuccessfullyRenamedB"))) RenamedAgainInAPINotesB {
// CHECK: typedef enum __attribute__((swift_name("SuccessfullyRenamedC"))) {
// CHECK-NEXT: kConstantInAnonEnum
// CHECK-NEXT: } AnonEnumWithTypedefName

void test(OverriddenTypes *overridden) {
  int *ip1 = global_int_ptr; // expected-warning{{incompatible pointer types initializing 'int *' with an expression of type 'double (*)(int, int)'}}

  int *ip2 = global_int_fun( // expected-warning{{incompatible pointer types initializing 'int *' with an expression of type 'char *'}}
               ip2, // expected-warning{{incompatible pointer types passing 'int *' to parameter of type 'double *'}}
               ip2); // expected-warning{{incompatible pointer types passing 'int *' to parameter of type 'float *'}}

  int *ip3 = [overridden // expected-warning{{incompatible pointer types initializing 'int *' with an expression of type 'char *'}}
                methodToMangle: ip3 // expected-warning{{incompatible pointer types sending 'int *' to parameter of type 'double *'}}
                        second: ip3]; // expected-warning{{incompatible pointer types sending 'int *' to parameter of type 'float *'}}

  int *ip4 = overridden.intPropertyToMangle; // expected-warning{{incompatible pointer types initializing 'int *' with an expression of type 'double *'}}
}

// expected-note@SomeKit/SomeKit.h:42{{passing argument to parameter 'ptr' here}}
// expected-note@SomeKit/SomeKit.h:42{{passing argument to parameter 'ptr2' here}}
// expected-note@SomeKit/SomeKit.h:48{{passing argument to parameter 'ptr1' here}}
// expected-note@SomeKit/SomeKit.h:48{{passing argument to parameter 'ptr2' here}}

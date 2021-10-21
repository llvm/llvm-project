// Test this without pch.
// RUN: %clang_cc1 -fblocks -include %S/objc_exprs.h -fsyntax-only -verify %s

// Test with pch.
// RUN: %clang_cc1 -x objective-c-header -emit-pch -fblocks -o %t %S/objc_exprs.h
// RUN: %clang_cc1 -fblocks -include-pch %t -fsyntax-only -verify %s 

// Expressions
int *A1 = (objc_string)0;   // expected-warning {{aka 'NSString *'}}

char A2 = (objc_encode){};  // expected-error {{not a compile-time constant}} \
                               expected-warning {{char[2]}}

int *A3 = (objc_protocol)0; // expected-warning {{aka 'Protocol *'}}


// Types.
int *T0 = (objc_id_protocol_ty)0; // expected-warning {{aka 'id<foo>'}}

int *T1 = (objc_interface_ty)0; // expected-warning {{aka 'itf *'}}
int *T2 = (objc_qual_interface_ty)0; // expected-warning {{aka 'itf<foo> *'}}

objc_selector_noArgs s1;
objc_selector_oneArg s2;
objc_selector_twoArg s3;



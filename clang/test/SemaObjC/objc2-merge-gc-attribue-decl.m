// RUN: %clang_cc1 -triple i386-apple-darwin9 -fobjc-gc -fsyntax-only -verify %s
@interface INTF @end

extern INTF* p2;
extern __strong INTF* p2;

extern __strong id p1;
extern id p1;

extern id CFRunLoopGetMain(void);
extern __strong id CFRunLoopGetMain(void);

extern __weak id WLoopGetMain(void); // expected-note {{previous declaration is here}}
extern id WLoopGetMain(void);	// expected-error {{conflicting types for 'WLoopGetMain'}}

extern id p3;	// expected-note {{previous declaration is here}}
extern __weak id p3;	// expected-error {{redeclaration of 'p3' with a different type}}

extern void *p4; // expected-note {{previous declaration is here}}
extern void * __strong p4; // expected-error {{redeclaration of 'p4' with a different type}}

extern id p5;
extern __strong id p5;

extern char* __strong p6; // expected-note {{previous declaration is here}}
extern char* p6; // expected-error {{redeclaration of 'p6' with a different type}}

extern __strong char* p7; // expected-note {{previous declaration is here}}
extern char* p7; // expected-error {{redeclaration of 'p7' with a different type}}

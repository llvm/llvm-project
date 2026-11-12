// RUN: %clang_cc1 -Wduplicate-method-match -fsyntax-only -verify %s

@protocol P1 @end
@protocol P2 @end
@protocol P3 @end

@interface INTF 
- (INTF*) METH1;	// expected-note {{previous declaration is here}}
- (INTF<P1>*) METH1;	// expected-error {{duplicate declaration of method 'METH1'}}

- (INTF<P2,P1>*) METH2;  // expected-note {{previous declaration is here}}
- (INTF<P2,P1,P3>*) METH2;  // expected-error {{duplicate declaration of method 'METH2'}}

- (INTF<P2,P1,P3>*) METH3; // expected-note {{previous declaration is here}}
- (INTF<P3,P1,P2, P3>*) METH3; // expected-warning {{multiple declarations of method 'METH3' found and ignored}}

@end

INTF<P2,P1,P3>* p1;


// RUN: %clang_cc1 -fsyntax-only -verify %s

@compatibility_alias alias4 foo; // expected-warning {{cannot find interface declaration for 'foo'}}

@class class2; // expected-note {{previous declaration is here}}
@class class3;

typedef int I;  // expected-note {{previous declaration is here}}

@compatibility_alias alias1 I;  // expected-warning {{cannot find interface declaration for 'I'}}

@compatibility_alias alias class2;     
@compatibility_alias alias class3;   // expected-error {{conflicting types for alias 'alias'}}


typedef int alias2;	// expected-note {{previous declaration is here}}
@compatibility_alias alias2 class3;  // expected-error {{conflicting types for alias 'alias2'}}

alias *p;
class2 *p2;

int foo (void)
{

	if (p == p2) {
	  int alias = 1;
	}

	alias *p3;
	return p3 == p2;
}

// RUN: %clang_cc1 -fsyntax-only -verify %s

struct S; // expected-note{{forward declaration of 'struct S'}}
typedef int FOO();

@interface INTF
{
	struct F {} JJ;
	int arr[];  // expected-error {{flexible array member 'arr' with type 'int[]' is not at the end of class}}
	struct S IC;  // expected-error {{field has incomplete type}}
	              // expected-note@-1 {{next instance variable declaration is here}}
	struct T { // expected-note {{previous definition is here}}
	  struct T {} X;  // expected-error {{nested redefinition of 'T'}}
	}YYY; 
	FOO    BADFUNC;  // expected-error {{field 'BADFUNC' declared as a function}}
	int kaka;	// expected-note {{previous declaration is here}}
	int kaka;	// expected-error {{duplicate member 'kaka'}}
	char ch[];
}
@end

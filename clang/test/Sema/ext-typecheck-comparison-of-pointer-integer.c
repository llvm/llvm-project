// RUN: %clang_cc1 -triple x86_64-apple-darwin -fsyntax-only  -verify -DEXPECTWARNING %s 
// RUN: %clang_cc1 -triple x86_64-apple-darwin -fsyntax-only  -verify -Wno-pointer-integer-compare %s 

#ifdef EXPECTWARNING
// expected-warning@+6 {{comparison between pointer and integer ('int' and 'int *')}}
#else
// expected-no-diagnostics 
#endif

int test_ext_typecheck_comparison_of_pointer_integer(int integer, int * pointer) {
	return integer != pointer; 
}

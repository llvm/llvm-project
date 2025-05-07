
// RUN: %clang_cc1 -isystem %S/mock-sdk -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -isystem %S/mock-sdk -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>
#include <extern-array-mock.h>

void bar(const unsigned	*pointer);

void foo(void){
	f();
	bar(externArray); // expected-warning{{accessing elements of an unannotated incomplete array always fails at runtime}}
	unsigned *ptr = externArray; // expected-warning{{accessing elements of an unannotated incomplete array always fails at runtime}}
}

extern const char baz[__null_terminated];

void qux(void) {
  const char *__null_terminated x = baz; // ok
}

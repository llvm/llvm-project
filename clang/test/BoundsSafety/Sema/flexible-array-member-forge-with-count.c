
// RUN: %clang_cc1 -fbounds-safety -verify %s -o /dev/null
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s -o /dev/null

#include <ptrcheck.h>

typedef struct {
	int len;
	unsigned fam[__counted_by(len)];
} S;

void bar(const unsigned	*pointer);

void foo(void){
	S s;
	bar(s.fam);
	unsigned *ptr = s.fam;
	ptr = __unsafe_forge_bidi_indexable(unsigned *, s.fam, s.len * sizeof(unsigned));
}

// expected-no-diagnostics

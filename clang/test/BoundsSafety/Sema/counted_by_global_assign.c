
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
#include <ptrcheck.h>

int len;
int *__counted_by(len) ptr;

void test_len_assign() {
  len = 0;
  // expected-error@-1{{assignment to 'len' requires corresponding assignment to 'int *__single __counted_by(len)' (aka 'int *__single') 'ptr'; add self assignment 'ptr = ptr' if the value has not changed}}
}

void test_ptr_assign(int *__bidi_indexable arg) {
  ptr = arg;
  // expected-error@-1{{assignment to 'int *__single __counted_by(len)' (aka 'int *__single') 'ptr' requires corresponding assignment to 'len'; add self assignment 'len = len' if the value has not changed}}
}

void test_assign(int *__bidi_indexable arg) {
  len = 1;
  ptr = arg;
}

extern int extlen;
extern int *__counted_by(extlen) extptr;

void test_extlen_assign() {
  extlen = 0;
  // expected-error@-1{{assignment to 'extlen' requires corresponding assignment to 'int *__single __counted_by(extlen)' (aka 'int *__single') 'extptr'; add self assignment 'extptr = extptr' if the value has not changed}}
}

void test_extptr_assign(int *__bidi_indexable arg) {
  extptr = arg;
  // expected-error@-1{{assignment to 'int *__single __counted_by(extlen)' (aka 'int *__single') 'extptr' requires corresponding assignment to 'extlen'; add self assignment 'extlen = extlen' if the value has not changed}}
}

void test_ext_assign(int *__bidi_indexable arg) {
  extlen = 1;
  extptr = arg;
}

int shared_len = 0;
int *__counted_by(shared_len) ptr2;
char *__counted_by(shared_len) ptr3;

void test_shared_len_assign() {
  // expected-error@+2{{assignment to 'shared_len' requires corresponding assignment to 'int *__single __counted_by(shared_len)' (aka 'int *__single') 'ptr2'; add self assignment 'ptr2 = ptr2' if the value has not changed}}
  // expected-error@+1{{assignment to 'shared_len' requires corresponding assignment to 'char *__single __counted_by(shared_len)' (aka 'char *__single') 'ptr3'; add self assignment 'ptr3 = ptr3' if the value has not changed}}
  shared_len = 0;
}

void test_ptr2_assign(int *__bidi_indexable arg) {
  // expected-error@+1{{assignment to 'int *__single __counted_by(shared_len)' (aka 'int *__single') 'ptr2' requires corresponding assignment to 'shared_len'; add self assignment 'shared_len = shared_len' if the value has not changed}}
  ptr2 = arg;
}

void test_ptr3_assign(char *__bidi_indexable arg) {
  // expected-error@+1{{assignment to 'char *__single __counted_by(shared_len)' (aka 'char *__single') 'ptr3' requires corresponding assignment to 'shared_len'; add self assignment 'shared_len = shared_len' if the value has not changed}}
  ptr3 = arg;
}

void test_shared_len_ptr2_assign(int *__bidi_indexable arg) {
  // expected-error@+1{{assignment to 'shared_len' requires corresponding assignment to 'char *__single __counted_by(shared_len)' (aka 'char *__single') 'ptr3'; add self assignment 'ptr3 = ptr3' if the value has not changed}}
  shared_len = 1;
  ptr2 = arg;
}

void test_shared_len_ptr3_assign(char *__bidi_indexable arg) {
  // expected-error@+1{{assignment to 'shared_len' requires corresponding assignment to 'int *__single __counted_by(shared_len)' (aka 'int *__single') 'ptr2'; add self assignment 'ptr2 = ptr2' if the value has not changed}}
  shared_len = 1;
  ptr3 = arg;
}


void test_shared_assign(int *__bidi_indexable arg1,
                        char *__bidi_indexable arg2) {
  shared_len = 1;
  ptr3 = arg2;
  ptr2 = arg1;
}

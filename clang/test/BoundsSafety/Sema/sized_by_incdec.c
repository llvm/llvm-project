
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
#include <ptrcheck.h>

void Foo(int *__sized_by(*len) *ptr, int *len) {
  (*ptr)++; // expected-note{{previously assigned here}}
  // expected-error@+2{{assignment to 'int *__single __sized_by(*len)' (aka 'int *__single') '*ptr' requires corresponding assignment to '*len'; add self assignment '*len = *len' if the value has not changed}}
  // expected-error@+1{{multiple consecutive assignments to a dynamic count pointer 'ptr' must be simplified; keep only one of the assignments}}
  ++(*ptr);
}

void FooOrNull(int *__sized_by_or_null(*len) *ptr, int *len) {
  (*ptr)++; // expected-note{{previously assigned here}}
  // expected-error@+2{{assignment to 'int *__single __sized_by_or_null(*len)' (aka 'int *__single') '*ptr' requires corresponding assignment to '*len'; add self assignment '*len = *len' if the value has not changed}}
  // expected-error@+1{{multiple consecutive assignments to a dynamic count pointer 'ptr' must be simplified; keep only one of the assignments}}
  ++(*ptr);
}

void Bar(int *__sized_by(*len) *ptr, int *len) {
  *ptr = 0;
  (*len)++;
}

void BarOrNull(int *__sized_by_or_null(*len) *ptr, int *len) {
  *ptr = 0;
  (*len)++;
}

void TestPtrPostIncrement(int *__sized_by(*len) *ptr, int *len) {
  (*ptr)++;
  (*len)--;
}

void TestPtrPostIncrementOrNull(int *__sized_by_or_null(*len) *ptr, int *len) {
  (*ptr)++;
  (*len)--;
}

void TestLenPostIncrementOrNull(int *__sized_by_or_null(*len) ptr, int *len) {
  ptr = ptr;
  (*len)++; // expected-error{{incrementing '*len' without updating 'ptr' always traps}}
}

void TestLenPostIncrement(int *__sized_by(*len) ptr, int *len) {
  ptr = ptr;
  (*len)++; // expected-error{{incrementing '*len' without updating 'ptr' always traps}}
}

void TestPtrPreDecrement(int *__sized_by(*len) *ptr, int *len) {
  --(*ptr); // expected-error{{negative pointer arithmetic on '__sized_by' pointer always traps}}
}

void TestPtrPreDecrementOrNull(int *__sized_by_or_null(*len) *ptr, int *len) {
  --(*ptr); // expected-error{{negative pointer arithmetic on '__sized_by_or_null' pointer always traps}}
}

typedef struct {
  char *__sized_by(len1 + len2) buf;
  unsigned len1;
  unsigned len2;
} S;

void TestMultipleCounts1(S *sp, char *__bidi_indexable new_ptr) {
  sp->buf = new_ptr;
  sp->len2 = sp->len2;
  sp->len1++;
}

void TestMultipleCounts2(S *sp, char *__bidi_indexable new_ptr) {
  sp->buf = new_ptr;
  sp->len2++;
  sp->len1 = sp->len1;
}

void TestMultipleCounts3(S *sp) {
  sp->buf = sp->buf;
  sp->len2++; // expected-error{{incrementing 'sp->len2' without updating 'sp->buf' always traps}}
  sp->len1 = sp->len1;
}

typedef struct {
  char *__sized_by_or_null(len1 + len2) buf;
  unsigned len1;
  unsigned len2;
} SOrNull;

void TestMultipleCounts1OrNull(SOrNull *sp, char *__bidi_indexable new_ptr) {
  sp->buf = new_ptr;
  sp->len2 = sp->len2;
  sp->len1++;
}

void TestMultipleCounts2OrNull(S *sp, char *__bidi_indexable new_ptr) {
  sp->buf = new_ptr;
  sp->len2++;
  sp->len1 = sp->len1;
}

void TestMultipleCounts3OrNull(S *sp) {
  sp->buf = sp->buf;
  sp->len2++; // expected-error{{incrementing 'sp->len2' without updating 'sp->buf' always traps}}
  sp->len1 = sp->len1;
}

typedef struct {
  char *__sized_by(len) buf;
  unsigned len;
} T;

void Baz(T *tp) {
  tp->buf = tp->buf;
  tp->len++; // expected-error{{incrementing 'tp->len' without updating 'tp->buf' always traps}}
}

void Qux(T *tp) {
  ++tp->len; // expected-error{{incrementing 'tp->len' without updating 'tp->buf' always traps}}
  tp->buf = tp->buf;
}

void Quux(T *tp) {
  tp->buf = tp->buf;
  tp->len--;
}

void Quuz(T *tp) {
  tp->len--; // expected-error{{assignment to 'tp->len' requires corresponding assignment to 'char *__single __sized_by(len)' (aka 'char *__single') 'tp->buf'; add self assignment 'tp->buf = tp->buf' if the value has not changed}}
}

void Corge(T *tp) {
  tp->len+=2; // expected-error{{assignment to 'tp->len' requires corresponding assignment to 'char *__single __sized_by(len)' (aka 'char *__single') 'tp->buf'; add self assignment 'tp->buf = tp->buf' if the value has not changed}}
}

typedef struct {
  char *__sized_by_or_null(len) buf;
  unsigned len;
} TOrNull;

void BazOrNull(TOrNull *tp) {
  tp->buf = tp->buf;
  tp->len++; // expected-error{{incrementing 'tp->len' without updating 'tp->buf' always traps}}
}

void QuxOrNull(TOrNull *tp) {
  ++tp->len; // expected-error{{incrementing 'tp->len' without updating 'tp->buf' always traps}}
  tp->buf = tp->buf;
}

void QuuxOrNull(TOrNull *tp) {
  tp->buf = tp->buf;
  tp->len--;
}

void QuuzOrNull(TOrNull *tp) {
  tp->len--; // expected-error{{assignment to 'tp->len' requires corresponding assignment to 'char *__single __sized_by_or_null(len)' (aka 'char *__single') 'tp->buf'; add self assignment 'tp->buf = tp->buf' if the value has not changed}}
}

void CorgeOrNull(TOrNull *tp) {
  tp->len+=2; // expected-error{{assignment to 'tp->len' requires corresponding assignment to 'char *__single __sized_by_or_null(len)' (aka 'char *__single') 'tp->buf'; add self assignment 'tp->buf = tp->buf' if the value has not changed}}
}

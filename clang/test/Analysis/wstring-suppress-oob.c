// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.cstring \
// RUN:   -analyzer-checker=unix.Malloc \
// RUN:   -analyzer-checker=alpha.unix.cstring.BufferOverlap \
// RUN:   -analyzer-checker=alpha.unix.cstring.NotNullTerminated \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false
//
// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -triple x86_64-pc-windows-msvc19.11.0 \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.cstring \
// RUN:   -analyzer-checker=unix.Malloc \
// RUN:   -analyzer-checker=alpha.unix.cstring.BufferOverlap \
// RUN:   -analyzer-checker=alpha.unix.cstring.NotNullTerminated \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false

typedef __SIZE_TYPE__ size_t;
typedef __WCHAR_TYPE__ wchar_t;

void clang_analyzer_eval(int);

void *malloc(size_t);
void free(void *);

wchar_t *wmemset(wchar_t *s, wchar_t c, size_t n);

size_t wcslen(const wchar_t *s);

void wmemset_char_malloc_overflow_with_nullchr_gives_unknown(void) {
  wchar_t *str = (wchar_t *)malloc(10 * sizeof(wchar_t));
  wmemset(str, '\0', 12);
  // If the `wmemset` doesn't set the whole buffer exactly,
  // then the buffer is invalidated by the checker.
  clang_analyzer_eval(str[1] == 0); // expected-warning{{UNKNOWN}}
  free(str);
}

void wmemset_char_array_set_wcslen(void) {
  wchar_t str[5] = L"abcd";
  clang_analyzer_eval(wcslen(str) == 4); // expected-warning{{TRUE}}
  wmemset(str, L'Z', 10);
  clang_analyzer_eval(str[0] != L'Z');     // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(wcslen(str) < 10);  // expected-warning{{FALSE}}
}

struct POD_wmemset {
  int num;
  wchar_t c;
};

void wmemset_struct_complete(void) {
  struct POD_wmemset pod;
  pod.num = 1;
  pod.c = L'A';
  wmemset((wchar_t*)&pod.num, 0, sizeof(struct POD_wmemset) / sizeof(wchar_t));

  clang_analyzer_eval(pod.num == 0);  // expected-warning{{TRUE}}
  clang_analyzer_eval(pod.c == '\0'); // expected-warning{{TRUE}}
}

void wmemset_struct_complete_incorrect_size(void) {
  struct POD_wmemset pod;
  pod.num = 1;
  pod.c = L'A';
  _Static_assert(sizeof(wchar_t) != sizeof(char), "Expected by this test case");
  wmemset((wchar_t*)&pod, 0, sizeof(struct POD_wmemset)); // count is off if wchar_t != char

  clang_analyzer_eval(pod.num == 0);  // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(pod.c == '\0'); // expected-warning{{UNKNOWN}}
}

void wmemset_struct_first_field_equivalent_to_complete(void) {
  struct POD_wmemset pod;
  pod.num = 1;
  pod.c = L'A';
  wmemset((wchar_t*)&pod.num, 0, sizeof(struct POD_wmemset) / sizeof(wchar_t));
  clang_analyzer_eval(pod.num == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(pod.c == 0);   // expected-warning{{TRUE}}
}

void wmemset_struct_second_field(void) {
  struct POD_wmemset pod;
  pod.num = 1;
  pod.c = L'A';
  wmemset((wchar_t*)&pod.c, 0, sizeof(struct POD_wmemset) / sizeof(wchar_t));
  // wmemset crosses the boundary of pod.c, so entire pod is invalidated.
  clang_analyzer_eval(pod.num == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(pod.c == 0);   // expected-warning{{UNKNOWN}}
}

void wmemset_struct_second_field_no_oob(void) {
  struct POD_wmemset pod;
  pod.num = 1;
  pod.c = L'A';
  wmemset((wchar_t*)&pod.c, 0, 1);
  // wmemset stays within pod.c, so pod.num is unaffected.
  clang_analyzer_eval(pod.num == 1); // expected-warning{{TRUE}}
  // pod.c is invalidated, while it should be set to 0.
  // limitation of current modeling.
  clang_analyzer_eval(pod.c == 0);   // expected-warning{{UNKNOWN}}
}

union U_wmemset {
  int i;
  double d;
  char c;
};

void wmemset_union_field(void) {
  union U_wmemset u;
  u.i = 5;
  wmemset((wchar_t*)&u.i, L'\0', sizeof(union U_wmemset));
  // Note: This should be TRUE, analyzer can't handle union perfectly now.
  clang_analyzer_eval(u.d == 0); // expected-warning{{UNKNOWN}}
}

void wmemset_len_nonexact_invalidate() {
  struct S {
    wchar_t array[10];
    int field;
  } s;
  s.array[0] = L'a';
  s.field = 1;
  clang_analyzer_eval(s.array[0] == L'a'); // expected-warning{{TRUE}}
  wmemset(s.array, L'\0', 5);
  // Invalidating the whole buffer because len does not match its full length
  clang_analyzer_eval(s.array[0] == L'\0'); // expected-warning{{UNKNOWN}}
  // wmemset stays within the bounds of s.array, so s.field is unaffected
  clang_analyzer_eval(s.field == 1); // expected-warning{{TRUE}}

  wmemset(s.array, L'\0', sizeof(s.array)); // length in bytes means it will actually overflow
  // Invalidating the whole buffer because len does not match its full length
  clang_analyzer_eval(s.array[0] == L'\0'); // expected-warning{{UNKNOWN}}
  // wmemset overflows the s.array buffer, so s.field is also invalidated
  clang_analyzer_eval(s.field == 1); // expected-warning{{UNKNOWN}}

  s.array[0] = L'a';
  s.field = 1;

  wmemset(s.array, L'\0', sizeof(s.array) / sizeof(wchar_t));
  // Modeling limitation: wmemset clears exactly s.array,
  // but not entire s, so s.array is invalidated instead of being set to 0.
  clang_analyzer_eval(s.array[0] == L'\0'); // expected-warning{{UNKNOWN}}
  // wmemset stays within the bounds of s.array, so s.field is preserved
  clang_analyzer_eval(s.field == 1); // expected-warning{{TRUE}}
}
wchar_t* wcsncpy(wchar_t *restrict s1, const wchar_t *restrict s2, size_t n);

// Make sure the checker does not crash when the length argument is way beyond the
// extents of the source and dest arguments
void wcsncpy_cstringchecker_bounds_nocrash(void) {
  wchar_t *p = malloc(2 * sizeof(wchar_t));
  // sizeof(L"AAA") returns 4*sizeof(wchar_t), e.g., 16, which is longer than
  // the number of characters in L"AAA" - 4:
  wcsncpy(p, L"AAA", sizeof(L"AAA"));
  free(p);
}

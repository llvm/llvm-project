// RUN: %clang_analyze_cc1 %s -std=c89 \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.StdCLibraryFunctions \
// RUN:   -analyzer-output=text \
// RUN:   -triple x86_64-pc-windows-msvc -verify=expected,applied

// RUN: %clang_analyze_cc1 %s -std=c99 \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.StdCLibraryFunctions \
// RUN:   -analyzer-output=text \
// RUN:   -triple x86_64-pc-windows-msvc -verify=expected,applied

// RUN: %clang_analyze_cc1 %s -std=c11 \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.StdCLibraryFunctions \
// RUN:   -analyzer-output=text \
// RUN:   -triple x86_64-pc-windows-msvc -verify=expected,applied

// The MSVC C runtime (UCRT) declares fopen without the restrict qualifiers of
// the standard prototype. The summary of fopen is still applied, so the path
// has the note about the successful call in all language modes.
// See std-c-library-functions-restrict-msvc.c.

typedef struct _iobuf FILE;
FILE *fopen(char const *_FileName, char const *_Mode);

void test_fopen_note(void) {
  int *p = 0; // expected-note{{'p' initialized to a null pointer value}}
  FILE *fp = fopen("file", "r"); // applied-note{{Assuming that 'fopen' is successful}}
  if (fp) // applied-note{{'fp' is non-null}}
          // not-applied-note@-1{{Assuming 'fp' is non-null}}
          // expected-note@-2{{Taking true branch}}
    *p = 1; // expected-warning{{Dereference of null pointer (loaded from variable 'p')}}
            // expected-note@-1{{Dereference of null pointer (loaded from variable 'p')}}
}

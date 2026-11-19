// RUN: %clang_analyze_cc1 %s -std=c89 \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.StdCLibraryFunctions \
// RUN:   -triple x86_64-pc-windows-msvc -verify

// RUN: %clang_analyze_cc1 %s -std=c99 \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.StdCLibraryFunctions \
// RUN:   -triple x86_64-pc-windows-msvc -verify

// RUN: %clang_analyze_cc1 %s -std=c11 \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.StdCLibraryFunctions \
// RUN:   -triple x86_64-pc-windows-msvc -verify

// RUN: %clang_analyze_cc1 %s -std=c99 \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.StdCLibraryFunctions \
// RUN:   -triple x86_64-unknown-linux-gnu -verify

// RUN: %clang_analyze_cc1 %s -std=c99 \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.StdCLibraryFunctions \
// RUN:   -triple x86_64-w64-windows-gnu -verify

// The UCRT declares fopen without restrict. Its summary should apply on all
// targets, regardless of whether the declaration uses restrict.

typedef struct _iobuf FILE;
FILE *fopen(char const *_FileName, char const *_Mode);

void test_fopen_null_filename(void) {
  fopen(0, "r"); // expected-warning{{The 1st argument to 'fopen' is NULL but should not be NULL}}
}

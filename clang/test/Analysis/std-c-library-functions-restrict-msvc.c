// RUN: %clang_analyze_cc1 %s -std=c89 \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.StdCLibraryFunctions \
// RUN:   -triple x86_64-pc-windows-msvc -verify=applied

// RUN: %clang_analyze_cc1 %s -std=c99 \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.StdCLibraryFunctions \
// RUN:   -triple x86_64-pc-windows-msvc -verify=not-applied

// RUN: %clang_analyze_cc1 %s -std=c11 \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.StdCLibraryFunctions \
// RUN:   -triple x86_64-pc-windows-msvc -verify=not-applied

// RUN: %clang_analyze_cc1 %s -std=c99 \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.StdCLibraryFunctions \
// RUN:   -triple x86_64-unknown-linux-gnu -verify=not-applied

// RUN: %clang_analyze_cc1 %s -std=c99 \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.StdCLibraryFunctions \
// RUN:   -triple x86_64-w64-windows-gnu -verify=not-applied

// The MSVC C runtime (UCRT) declares fopen without the restrict qualifiers of
// the standard prototype. In C89, the summary has no restrict either, so it is
// applied. From C99 on, the summary requires restrict and does not match, so
// the summary is not applied to the fopen of MSVC.
//
// FIXME: The summary should be applied in C99 and later too when targeting the
// MSVC environment. Its C runtime does not follow the standard prototype, but
// this declaration is still the fopen of the C library.
//
// On other targets (including MinGW, which has its own C library headers), the
// C library is expected to follow the standard prototype, so the summary is
// rightly not applied to a declaration without restrict.

// not-applied-no-diagnostics

typedef struct _iobuf FILE;
FILE *fopen(char const *_FileName, char const *_Mode);

void test_fopen_null_filename(void) {
  fopen(0, "r"); // applied-warning{{The 1st argument to 'fopen' is NULL but should not be NULL}}
}

// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.StdCLibraryFunctions \
// RUN:   -analyzer-checker=debug.StdCLibraryFunctionsTester \
// RUN:   -analyzer-config unix.StdCLibraryFunctions:DisplayLoadedSummaries=true \
// RUN:   -triple i686-unknown-linux 2>&1 | FileCheck %s

// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.StdCLibraryFunctions \
// RUN:   -analyzer-checker=debug.StdCLibraryFunctionsTester \
// RUN:   -analyzer-config unix.StdCLibraryFunctions:DisplayLoadedSummaries=true \
// RUN:   -triple x86_64-pc-windows-msvc 2>&1 | FileCheck %s --check-prefix=MSVC

// The signatures for these functions are the same and they specify their
// parameter with the restrict qualifier. In C, the signature should match only
// if the restrict qualifier is there on the parameter. Thus, the summary
// should be loaded for the last two declarations only.
// FIXME: The MSVC environment should be an exception: the Microsoft C runtime
// omits the restrict qualifiers of the standard prototypes, so the signature
// should match regardless of the restrict qualifier, and the summary should be
// loaded for all three declarations.
void __test_restrict_param_0(void *p);
void __test_restrict_param_1(void *__restrict p);
void __test_restrict_param_2(void *restrict p);

// CHECK-NOT: Loaded summary for: void __test_restrict_param_0
//     CHECK: Loaded summary for: void __test_restrict_param_1(void *restrict p)
//     CHECK: Loaded summary for: void __test_restrict_param_2(void *restrict p)

// MSVC-NOT: Loaded summary for: void __test_restrict_param_0
//     MSVC: Loaded summary for: void __test_restrict_param_1(void *restrict p)
//     MSVC: Loaded summary for: void __test_restrict_param_2(void *restrict p)

// Must have at least one call expression to initialize the summary map.
int bar(void);
void foo(void) {
  bar();
}

// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.StdCLibraryFunctions \
// RUN:   -analyzer-checker=debug.StdCLibraryFunctionsTester \
// RUN:   -analyzer-config unix.StdCLibraryFunctions:DisplayLoadedSummaries=true \
// RUN:   -triple i686-unknown-linux 2>&1 | FileCheck %s

// The signature should match regardless of the restrict qualifier, including
// the __restrict extension in C++.
void __test_restrict_param_0(void *p);
void __test_restrict_param_1(void *__restrict p);
// The below declaration is illegal, "restrict" is not a keyword in C++.
// void __test_restrict_param_2(void *restrict p);

// CHECK: Loaded summary for: void __test_restrict_param_0(void *p)
// CHECK: Loaded summary for: void __test_restrict_param_1(void *__restrict p)

// Must have at least one call expression to initialize the summary map.
int bar(void);
void foo() {
  bar();
}

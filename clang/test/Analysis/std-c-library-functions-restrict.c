// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=apiModeling.StdCLibraryFunctions \
// RUN:   -analyzer-checker=debug.StdCLibraryFunctionsTester \
// RUN:   -analyzer-config apiModeling.StdCLibraryFunctions:DisplayLoadedSummaries=true \
// RUN:   -triple i686-unknown-linux 2>&1 | FileCheck %s

// The signatures for these functions are the same and they specify their
// parameter with the restrict qualifier. In C, the signature should match only
// if the restrict qualifier is there on the parameter. Thus, the summary
// should be loaded for the last two declarations only.
void __test_restrict_param_0(void *p);
void __test_restrict_param_1(void *__restrict p);
void __test_restrict_param_2(void *restrict p);

// CHECK-NOT: Loaded summary for: void __test_restrict_param_0
//     CHECK: Loaded summary for: void __test_restrict_param_1(void *restrict p)
//     CHECK: Loaded summary for: void __test_restrict_param_2(void *restrict p)

// Must have at least one call expression to initialize the summary map.
int bar(void);
void foo() {
  bar();
}

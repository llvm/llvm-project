// Check that the more specific checkers report and not the generic
// StdCLibraryFunctionArgs checker.

// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=apiModeling.StdCLibraryFunctions \
// RUN:   -analyzer-config apiModeling.StdCLibraryFunctions:ModelPOSIX=true \
// RUN:   -analyzer-checker=alpha.unix.StdCLibraryFunctionArgs \
// RUN:   -analyzer-checker=alpha.unix.Stream \
// RUN:   -triple x86_64-unknown-linux-gnu \
// RUN:   -verify


// Make sure that all used functions have their summary loaded.

// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=apiModeling.StdCLibraryFunctions \
// RUN:   -analyzer-config apiModeling.StdCLibraryFunctions:ModelPOSIX=true \
// RUN:   -analyzer-checker=alpha.unix.StdCLibraryFunctionArgs \
// RUN:   -analyzer-checker=alpha.unix.Stream \
// RUN:   -analyzer-config apiModeling.StdCLibraryFunctions:DisplayLoadedSummaries=true \
// RUN:   -triple x86_64-unknown-linux 2>&1 | FileCheck %s

// CHECK: Loaded summary for: int isalnum(int)
// CHECK: Loaded summary for: unsigned long fread(void *restrict, size_t, size_t, FILE *restrict) __attribute__((nonnull(1)))
// CHECK: Loaded summary for: int fileno(FILE *stream)

void initializeSummaryMap();
// We analyze this function first, and the call expression inside initializes
// the summary map. This way we force the loading of the summaries. The
// summaries would not be loaded without this because during the first bug
// report in WeakDependency::checkPreCall we stop further evaluation. And
// StdLibraryFunctionsChecker lazily initializes its summary map from its
// checkPreCall.
void analyzeThisFirst() {
  initializeSummaryMap();
}

typedef __typeof(sizeof(int)) size_t;
struct FILE;
typedef struct FILE FILE;

int isalnum(int);
size_t fread(void *restrict, size_t, size_t, FILE *restrict) __attribute__((nonnull(1)));
int fileno(FILE *stream);

void test_uninit_arg() {
  int v;
  int r = isalnum(v); // \
  // expected-warning{{1st function call argument is an uninitialized value [core.CallAndMessage]}}
  (void)r;
}

void test_notnull_arg(FILE *F) {
  int *p = 0;
  fread(p, sizeof(int), 5, F); // \
  expected-warning{{Null pointer passed to 1st parameter expecting 'nonnull' [core.NonNullParamChecker]}}
}

void test_notnull_stream_arg() {
  fileno(0); // \
  // expected-warning{{Stream pointer might be NULL [alpha.unix.Stream]}}
}

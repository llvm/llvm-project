// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=alpha.unix.StdCLibraryFunctions \
// RUN:   -analyzer-config alpha.unix.StdCLibraryFunctions:ModelPOSIX=true \
// RUN:   -analyzer-config alpha.unix.StdCLibraryFunctions:DisplayLoadedSummaries=true \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -triple i686-unknown-linux 2>&1 | FileCheck %s --allow-empty

// We test here the implementation of our summary API with Optional types. In
// this TU we do not provide declaration for any of the functions that have
// summaries. The implementation should be able to handle the nonexistent
// declarations in a way that the summary is not added to the map. We expect no
// crashes (i.e. no optionals should be 'dereferenced') and no output.

// Must have at least one call expression to initialize the summary map.
int bar(void);
void foo(void) {
  bar();
}

// CHECK-NOT: Loaded summary for:

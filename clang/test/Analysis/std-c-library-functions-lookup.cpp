// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.StdCLibraryFunctions \
// RUN:   -analyzer-config unix.StdCLibraryFunctions:DisplayLoadedSummaries=true \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -triple i686-unknown-linux 2>&1 | FileCheck %s

//      CHECK: Loaded summary for: size_t fread(void *, size_t, size_t, FILE *)
//  CHECK-NOT: Loaded summary for: size_t fread(void *, size_t, size_t, MyFile *)

typedef unsigned int size_t;
typedef struct FILE FILE;
size_t fread(void *, size_t, size_t, FILE *);

struct MyFile;
size_t fread(void *, size_t, size_t, MyFile *);

// Must have at least one call expression to initialize the summary map.
int bar(void);
void foo() {
  bar();
}

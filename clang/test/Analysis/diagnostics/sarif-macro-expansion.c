// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Malloc \
// RUN:   -analyzer-output=sarif -verify %s -o - | FileCheck %s

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);

#define ALLOC int *x = (int *)malloc(12);

void ends_inside_expansion(void) {
  ALLOC // no-crash
} // expected-warning {{Potential leak of memory pointed to by 'x'}}

// The note covers the 'ALLOC' use rather than reaching into the macro body.
// CHECK:            "text": "Memory is allocated"
// CHECK:            "region": {
// CHECK-NEXT:         "endColumn": 8,
// CHECK-NEXT:         "endLine": [[#ALLOC_LINE:]],
// CHECK-NEXT:         "startColumn": 3,
// CHECK-NEXT:         "startLine": [[#ALLOC_LINE]]

#define IS_NULL !p

void ends_at_end_of_expansion(int *p) {
  if (IS_NULL)
    *p = 1; // expected-warning {{Dereference of null pointer}}
}

// This range already ended at the end of the expansion, so it worked before.
// CHECK:            "text": "Assuming 'p' is null"
// CHECK:            "region": {
// CHECK-NEXT:         "endColumn": 14,
// CHECK-NEXT:         "endLine": [[#IS_NULL_LINE:]],
// CHECK-NEXT:         "startColumn": 7,
// CHECK-NEXT:         "startLine": [[#IS_NULL_LINE]]


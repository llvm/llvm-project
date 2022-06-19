// RUN: %clang_analyze_cc1 -triple x86_64-linux-gnu -analyzer-checker=core,unix,debug.ExprInspection %s -verify -std=c90
// RUN: %clang_analyze_cc1 -triple x86_64-linux-gnu -analyzer-checker=core,unix,debug.ExprInspection %s -verify -std=c99
// RUN: %clang_analyze_cc1 -triple x86_64-linux-gnu -analyzer-checker=core,unix,debug.ExprInspection %s -verify -std=c11
// RUN: %clang_analyze_cc1 -triple x86_64-linux-gnu -analyzer-checker=core,unix,debug.ExprInspection %s -verify -std=c17

// RUN: %clang_analyze_cc1 -triple x86_64-linux-gnu -analyzer-checker=core,unix,debug.ExprInspection %s -verify -std=c++98 -x c++
// RUN: %clang_analyze_cc1 -triple x86_64-linux-gnu -analyzer-checker=core,unix,debug.ExprInspection %s -verify -std=c++03 -x c++
// RUN: %clang_analyze_cc1 -triple x86_64-linux-gnu -analyzer-checker=core,unix,debug.ExprInspection %s -verify -std=c++11 -x c++
// RUN: %clang_analyze_cc1 -triple x86_64-linux-gnu -analyzer-checker=core,unix,debug.ExprInspection %s -verify -std=c++14 -x c++
// RUN: %clang_analyze_cc1 -triple x86_64-linux-gnu -analyzer-checker=core,unix,debug.ExprInspection %s -verify -std=c++17 -x c++

// RUN: %clang_analyze_cc1 -triple x86_64-linux-gnu -analyzer-checker=core,unix,debug.ExprInspection %s -verify -std=c17 \
// RUN:    -analyzer-config consider-single-element-arrays-as-flexible-array-members=true -DSINGLE_ELEMENT_FAMS
// RUN: %clang_analyze_cc1 -triple x86_64-linux-gnu -analyzer-checker=core,unix,debug.ExprInspection %s -verify -std=c++17 -x c++ \
// RUN:    -analyzer-config consider-single-element-arrays-as-flexible-array-members=true -DSINGLE_ELEMENT_FAMS

typedef __typeof(sizeof(int)) size_t;
size_t clang_analyzer_getExtent(void *);
void clang_analyzer_dump(size_t);

void *alloca(size_t size);
void *malloc(size_t size);
void free(void *ptr);

void test_incomplete_array_fam(void) {
  typedef struct FAM {
    char c;
    int data[];
  } FAM;

  FAM fam;
  clang_analyzer_dump(clang_analyzer_getExtent(&fam));
  clang_analyzer_dump(clang_analyzer_getExtent(fam.data));
  // expected-warning@-2 {{4 S64b}}
  // expected-warning@-2 {{Unknown}}

  FAM *p = (FAM *)alloca(sizeof(FAM));
  clang_analyzer_dump(clang_analyzer_getExtent(p));
  clang_analyzer_dump(clang_analyzer_getExtent(p->data));
  // expected-warning@-2 {{4 U64b}}
  // expected-warning@-2 {{Unknown}}

  FAM *q = (FAM *)malloc(sizeof(FAM));
  clang_analyzer_dump(clang_analyzer_getExtent(q));
  clang_analyzer_dump(clang_analyzer_getExtent(q->data));
  // expected-warning@-2 {{4 U64b}}
  // expected-warning@-2 {{Unknown}}
  free(q);
}

void test_zero_length_array_fam(void) {
  typedef struct FAM {
    char c;
    int data[0];
  } FAM;

  FAM fam;
  clang_analyzer_dump(clang_analyzer_getExtent(&fam));
  clang_analyzer_dump(clang_analyzer_getExtent(fam.data));
  // expected-warning@-2 {{4 S64b}}
  // expected-warning@-2 {{Unknown}}

  FAM *p = (FAM *)alloca(sizeof(FAM));
  clang_analyzer_dump(clang_analyzer_getExtent(p));
  clang_analyzer_dump(clang_analyzer_getExtent(p->data));
  // expected-warning@-2 {{4 U64b}}
  // expected-warning@-2 {{Unknown}}

  FAM *q = (FAM *)malloc(sizeof(FAM));
  clang_analyzer_dump(clang_analyzer_getExtent(q));
  clang_analyzer_dump(clang_analyzer_getExtent(q->data));
  // expected-warning@-2 {{4 U64b}}
  // expected-warning@-2 {{Unknown}}
  free(q);
}

void test_single_element_array_possible_fam(void) {
  typedef struct FAM {
    char c;
    int data[1];
  } FAM;

#ifdef SINGLE_ELEMENT_FAMS
  FAM likely_fam;
  clang_analyzer_dump(clang_analyzer_getExtent(&likely_fam));
  clang_analyzer_dump(clang_analyzer_getExtent(likely_fam.data));
  // expected-warning@-2 {{8 S64b}}
  // expected-warning@-2 {{Unknown}}

  FAM *p = (FAM *)alloca(sizeof(FAM));
  clang_analyzer_dump(clang_analyzer_getExtent(p));
  clang_analyzer_dump(clang_analyzer_getExtent(p->data));
  // expected-warning@-2 {{8 U64b}}
  // expected-warning@-2 {{Unknown}}

  FAM *q = (FAM *)malloc(sizeof(FAM));
  clang_analyzer_dump(clang_analyzer_getExtent(q));
  clang_analyzer_dump(clang_analyzer_getExtent(q->data));
  // expected-warning@-2 {{8 U64b}}
  // expected-warning@-2 {{Unknown}}
  free(q);
#else
  FAM likely_fam;
  clang_analyzer_dump(clang_analyzer_getExtent(&likely_fam));
  clang_analyzer_dump(clang_analyzer_getExtent(likely_fam.data));
  // expected-warning@-2 {{8 S64b}}
  // expected-warning@-2 {{4 S64b}}

  FAM *p = (FAM *)alloca(sizeof(FAM));
  clang_analyzer_dump(clang_analyzer_getExtent(p));
  clang_analyzer_dump(clang_analyzer_getExtent(p->data));
  // expected-warning@-2 {{8 U64b}}
  // expected-warning@-2 {{4 S64b}}

  FAM *q = (FAM *)malloc(sizeof(FAM));
  clang_analyzer_dump(clang_analyzer_getExtent(q));
  clang_analyzer_dump(clang_analyzer_getExtent(q->data));
  // expected-warning@-2 {{8 U64b}}
  // expected-warning@-2 {{4 S64b}}
  free(q);
#endif
}

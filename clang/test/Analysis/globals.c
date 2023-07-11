// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s

typedef typeof(sizeof(int)) size_t;
size_t clang_analyzer_getExtent(const void *p);
void clang_analyzer_dump(size_t n);

extern const unsigned char extern_redecl[];
const unsigned char extern_redecl[] = { 1,2,3,4 };
const unsigned char tentative_redecl[];
const unsigned char tentative_redecl[] = { 1,2,3,4 };

const unsigned char direct_decl[] = { 1,2,3,4 };

void test_redeclaration_extent(void) {
  clang_analyzer_dump(clang_analyzer_getExtent(direct_decl));      // expected-warning {{4 S64b}}
  clang_analyzer_dump(clang_analyzer_getExtent(extern_redecl));    // expected-warning {{4 S64b}}
  clang_analyzer_dump(clang_analyzer_getExtent(tentative_redecl)); // expected-warning {{4 S64b}}
}

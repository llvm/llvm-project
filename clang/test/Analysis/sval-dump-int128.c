// RUN: %clang_analyze_cc1 -triple x86_64-linux-gnu -analyzer-checker=debug.ExprInspection %s -verify

void clang_analyzer_dump(unsigned __int128 x);

void testDumpInt128() {
  clang_analyzer_dump((unsigned __int128)5 << 64); // expected-warning{{92233720368547758080 U128b}}
}

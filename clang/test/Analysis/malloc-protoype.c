// RUN: %clang_analyze_cc1 -w -analyzer-checker=core,unix.Malloc -verify %s
// expected-no-diagnostics

// Test that strange prototypes doesn't crash the analyzer

void malloc(int i);
void valloc(int i);

void test1(void)
{
  malloc(1);
}

void test2(void)
{
  valloc(1);
}

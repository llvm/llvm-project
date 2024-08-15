// RUN: %clang_analyze_cc1 -triple=x86_64-unknown-unknown \
// RUN:      -analyzer-checker debug.ExprInspection,core -fheinous-gnu-extensions -w %s -verify

int clang_analyzer_eval(int);

int global;
void testRValueOutput() {
  int &ref = global;
  ref = 1;
  __asm__("" : "=r"(((int)(global))));  // don't crash on rvalue output operand
  clang_analyzer_eval(global == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(ref == 1);    // expected-warning{{UNKNOWN}}
}

void *MyMemcpy(void *d, const void *s, const int n) {
  asm volatile (
    "cld\n rep movsb\n"
    :: "S" (s), "D" (d), "c" (n) : "memory"
  );
  return d;
}

void testInlineAsmMemcpy(void)
{
    int a, b = 10, c;
    MyMemcpy(&a, &b, sizeof(b));
    c = a; // no-warning
}

void testInlineAsmMemcpyArray(void)
{
    int a[10], b[10] = {}, c;
    MyMemcpy(&a, &b, sizeof(b));
    c = a[8]; // no-warning
}

void testInlineAsmMemcpyUninit(void)
{
    int a[10], b[10] = {}, c;
    MyMemcpy(&a[1], &b[1], sizeof(b) - sizeof(b[1]));
    c = a[0]; // expected-warning{{Assigned value is garbage or undefined}}
}

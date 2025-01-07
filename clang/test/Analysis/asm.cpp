// RUN: %clang_analyze_cc1 -triple=x86_64-unknown-unknown \
// RUN:      -analyzer-checker debug.ExprInspection,core -Wno-error=invalid-gnu-asm-cast -w %s -verify

int clang_analyzer_eval(int);
void clang_analyzer_dump(int);
void clang_analyzer_dump_ptr(void *);

int global;
void testRValueOutput() {
  int origVal = global;
  __asm__("" : "=r"(((int)(global))));  // don't crash on rvalue output operand
  int newVal = global; // Value "after" the invalidation.
  clang_analyzer_eval(origVal == newVal); // expected-warning{{TRUE}} expected-warning{{FALSE}}
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
    c = a[0]; // FIXME: should be warning about uninitialized value, but invalidateRegions() also
              // invalidates super region.
}

void testInlineAsmMemcpyUninitLoop(const void *src, unsigned long len)
{
    int a[10], c;
    unsigned long toCopy = sizeof(a) < len ? sizeof(a) : len;

    MyMemcpy(a, src, toCopy);

    // Use index 1, since before use of invalidateRegions in VisitGCCAsmStmt, engine bound unknown SVal only to
    // first element.
    c = a[1]; // no-warning
}

void testAsmWithVoidPtrArgument()
{
  extern void *globalVoidPtr;
  clang_analyzer_dump(*(int *)globalVoidPtr); // expected-warning-re {{reg_${{[0-9]+}}<int Element{SymRegion{reg_${{[0-9]+}}<void * globalVoidPtr>},0 S64b,int}>}}
  clang_analyzer_dump_ptr(globalVoidPtr); // expected-warning-re {{&SymRegion{reg_${{[0-9]+}}<void * globalVoidPtr>}}}
  asm ("" : : "a"(globalVoidPtr)); // no crash
  clang_analyzer_dump(*(int *)globalVoidPtr); // expected-warning {{derived_}}
  clang_analyzer_dump_ptr(globalVoidPtr); // expected-warning-re {{&SymRegion{reg_${{[0-9]+}}<void * globalVoidPtr>}}}
}

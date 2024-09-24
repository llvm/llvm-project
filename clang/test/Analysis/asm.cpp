// RUN: %clang_analyze_cc1 -triple=x86_64-unknown-unknown \
// RUN:      -analyzer-checker debug.ExprInspection,core -Wno-error=invalid-gnu-asm-cast -w %s -verify

int clang_analyzer_eval(int);
void clang_analyzer_dump(int);
void clang_analyzer_dump_ptr(void *);

int global;
void testRValueOutput() {
  int &ref = global;
  ref = 1;
  __asm__("" : "=r"(((int)(global))));  // don't crash on rvalue output operand
  clang_analyzer_eval(global == 1); // expected-warning{{FALSE}}
                                    // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(ref == 1);    // expected-warning{{FALSE}}
                                    // expected-warning@-1{{TRUE}}
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

    for (unsigned long i = 0; i < toCopy; ++i)
      c = a[i]; // no-warning
}

void testAsmWithVoidPtrArgument()
{
  extern void *globalVoidPtr;
  clang_analyzer_dump(*(int *)globalVoidPtr); // expected-warning-re {{reg_${{[0-9]+}}<int Element{SymRegion{reg_${{[0-9]+}}<void * globalVoidPtr>},0 S64b,int}>}}
  clang_analyzer_dump_ptr(globalVoidPtr); // expected-warning-re {{&SymRegion{reg_${{[0-9]+}}<void * globalVoidPtr>}}}
  asm ("" : : "a"(globalVoidPtr)); // no crash
  clang_analyzer_dump(*(int *)globalVoidPtr); // expected-warning-re {{derived_$3{conj_$2{int, LC1, S{{[0-9]+}}, #1},Element{SymRegion{reg_$0<void * globalVoidPtr>},0 S64b,int}}}}
  clang_analyzer_dump_ptr(globalVoidPtr); // expected-warning-re {{&SymRegion{reg_${{[0-9]+}}<void * globalVoidPtr>}}}
}

// RUN: %clang_analyze_cc1 -analyzer-checker=core,osx -analyzer-output=text -verify %s

// expected-no-diagnostics

typedef void *CFTypeRef;
typedef struct _CFURLCacheRef *CFURLCacheRef;

CFTypeRef CustomCFRetain(CFTypeRef);
void invalidate(void *);
struct S1 {
  CFTypeRef s;
  CFTypeRef returnFieldAtPlus0() {
    return s;
  }
};
struct S2 {
  S1 *s1;
};
void foo(S1 *s1) {
  invalidate(s1);
  S2 s2;
  s2.s1 = s1;
  CustomCFRetain(s1->returnFieldAtPlus0());

  // Definitely no leak end-of-path note here. The retained pointer
  // is still accessible through s1 and s2.
  ((void) 0); // no-warning

  // FIXME: Ideally we need to warn after this invalidate(). The per-function
  // retain-release contract is violated: the programmer should release
  // the symbol after it was retained, within the same function.
  invalidate(&s2);
}

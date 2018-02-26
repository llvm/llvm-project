// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection,unix.Malloc -analyzer-config c++-allocator-inlining=true -analyzer-output=text -std=c++11 -verify %s

void clang_analyzer_eval(bool);

typedef __typeof__(sizeof(int)) size_t;

void *malloc(size_t size);

void *operator new(size_t size) throw() {
  void *x = malloc(size); // expected-note {{Memory is allocated}}
  if (!x) // expected-note    {{Assuming 'x' is non-null}}
          // expected-note@-1 {{Taking false branch}}
    return nullptr;
  return x;
}

void checkNewAndConstructorInlining() {
  int *s = new int; // expected-note   {{Calling 'operator new'}}
                    // expected-note@-1{{Returning from 'operator new'}}
} // expected-warning {{Potential leak of memory pointed to by 's'}}
  // expected-note@-1 {{Potential leak of memory pointed to by 's'}}

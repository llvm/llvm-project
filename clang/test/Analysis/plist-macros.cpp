// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix -analyzer-output=plist-multi-file %s -o %t.plist
// RUN: cat %t.plist | %diff_plist %S/Inputs/expected-plists/plist-macros.cpp.plist -


typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);

#define mallocmemory int *x = (int*)malloc(12);
void noteOnMacro(int y) {
  y++;
  y--;
  mallocmemory
  y++; 
  y++;
  delete x; // expected-warning {{Memory allocated by malloc() should be deallocated by free(), not 'delete'}}
}

void macroIsFirstInFunction(int y) {
  mallocmemory 
  y++; // expected-warning {{Potential leak of memory pointed to by 'x'}}
}

#define checkmacro p==0
void macroInExpressionAux(bool b);
int macroInExpression(int *p, int y) {;
  y++;
  macroInExpressionAux(checkmacro);

  return *p; // expected-warning {{Dereference of null pointer}}
}

#define noPathNoteMacro y+y
int macroInExpressionNoNote(int *p, int y) {;
  y++;
  if (5 + noPathNoteMacro)
    if (p)
      ;
  return *p; // expected-warning {{Dereference of null pointer}}
}

#define macroWithArg(mp) mp==0 
int macroWithArgInExpression(int *p, int y) {;
  y++;
  if (macroWithArg(p))
    ;
  return *p; // expected-warning {{Dereference of null pointer}}
}

#define multiNoteMacroWithError \
  if (p) \
    ;\
  *p = 5;
int useMultiNoteMacroWithError(int *p, int y) {;
  y++;
  multiNoteMacroWithError  // expected-warning {{Dereference of null pointer}}

  return *p;
}

#define multiNoteMacro \
if (p) \
  ;\
if (y) \
  ;
int useMultiNote(int *p, int y) {;
  y++;
  if (p) {}
  multiNoteMacro

  return *p; // expected-warning {{Dereference of null pointer}}
}

#define CALL_FN(a) null_deref(a)

void null_deref(int *a) {
  if (a)
    return;
  *a = 1; // expected-warning {{Dereference of null pointer}}
}

void test1() {
  CALL_FN(0);
}

void test2(int *p) {
  CALL_FN(p);
}

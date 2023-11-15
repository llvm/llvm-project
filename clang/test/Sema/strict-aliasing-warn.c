// RUN: %clang_cc1 %s -O0 -Wstrict-aliasing -S -o %t -verify=quiet
// RUN: %clang_cc1 %s -O2 -Wstrict-aliasing=0 -S -o %t -verify=quiet
// RUN: %clang_cc1 %s -O2 -Wno-strict-aliasing -S -o %t -verify=quiet
// RUN: %clang_cc1 %s -O2 -Wstrict-aliasing=1 -S -o %t -verify=level1,level12,level123
// RUN: %clang_cc1 %s -O2 -Wstrict-aliasing=2 -S -o %t -verify=level2,level23,level12,level123
// RUN: %clang_cc1 %s -O2 -Wstrict-aliasing=3 -S -o %t -verify=level23,level123
// RUN: %clang_cc1 %s -O2 -Wstrict-aliasing -S -o %t -verify=level23,level123
// RUN: %clang_cc1 %s -O2 -S -o %t -verify=level23,level123

// quiet-no-diagnostics

#if _LP64
// These names make more sense on an ilp32 machine
typedef long INT;
typedef long long LONG;
typedef unsigned long UINT;
#else
typedef int INT;
typedef long LONG;
typedef unsigned int UINT;
#endif
typedef short SHORT;

INT ScalarINT;
INT Ary[2];
struct {int m;} Struct;

_Complex int CPLx;

void ByVal(long long);
void ByPtr(void *);

void VarPtr(INT *Ptr) {
  // GCC: 1
  // level1-warning@+1{{type-punned pointer might break}}
  ByPtr((LONG *)(Ptr));
  // level1-note@-1{{not alias compatible}}

  // GCC:
  ByPtr((LONG *)((void *)(Ptr)));

  // GCC: 1
  // level1-warning@+1{{type-punned pointer might break}}
  ByVal(*(LONG *)(Ptr));
  // level1-note@-1{{not alias compatible}}
}

void Object() {
  // GCC: 1, 2
  // level2-warning@+2{{type-punned pointer breaks}}
  // level1-warning@+1{{type-punned pointer might break}}
  ByPtr((LONG *)(&ScalarINT));
  // level12-note@-1{{not alias compatible}}

  // GCC:
  ByPtr((LONG *)((void *)(&ScalarINT)));

  // GCC: 1, 2, 3
  // level23-warning@+2{{type-punned pointer breaks}}
  // level1-warning@+1{{type-punned pointer might break}}
  ByVal(*(LONG *)(&ScalarINT));
  // level123-note@-1{{not alias compatible}}
}

// Level 1, 2, 3 - 1, 2, 3
void DetectedVariants() {
  // GCC: 1, 2, 3
  // level23-warning@+2{{type-punned pointer breaks}}
  // level1-warning@+1{{type-punned pointer might break}}
  ByVal(*(LONG *)(&Ary[1]));
  // level123-note@-1{{not alias compatible}}

  // GCC: 1, 2, 3
  // level23-warning@+2{{type-punned pointer breaks}}
  // level1-warning@+1{{type-punned pointer might break}}
  ByVal(*(LONG *)(&Struct.m));
  // level123-note@-1{{not alias compatible}}

  // GCC: 1, 2, 3
  // level23-warning@+2{{type-punned pointer breaks}}
  // level1-warning@+1{{type-punned pointer might break}}
  ByVal(*(LONG *)(&(&Struct)->m));
  // level123-note@-1{{not alias compatible}}

  // GCC: 1, 2, 3
  // level23-warning@+2{{type-punned pointer breaks}}
  // level1-warning@+1{{type-punned pointer might break}}
  ByVal(*(LONG *)(&__real__(CPLx)));
  // level123-note@-1{{not alias compatible}}

  // GCC: 1, 2, 3
  // level23-warning@+2{{type-punned pointer breaks}}
  // level1-warning@+1{{type-punned pointer might break}}
  ByVal(*(LONG *)(&__imag__(CPLx)));
  // level123-note@-1{{not alias compatible}}
}

void Ok() {
  // GCC:
  ByPtr((UINT *)(&ScalarINT));
  // GCC:
  ByPtr((UINT *)((void *)(&ScalarINT)));
  // GCC:
  ByVal(*(UINT *)(&ScalarINT));
}

// Level 1, 2, 3 - 1, 2, 3
void Parens() {
  // GCC: 1, 2, 3
  // level23-warning@+2{{type-punned pointer breaks}}
  // level1-warning@+1{{type-punned pointer might break}}
  ByVal(*((LONG *)((&(ScalarINT)))));
  // level123-note@-1{{not alias compatible}}

  // GCC: 1, 2, 3
  // level23-warning@+2{{type-punned pointer breaks}}
  // level1-warning@+1{{type-punned pointer might break}}
  ByVal(*((LONG *)((&(Ary[1])))));
  // level123-note@-1{{not alias compatible}}
}

// Clang models may_alias as a decl attribute, not a type attribute.

typedef int MA __attribute__((may_alias));

void Frob(MA *a) {
  ByPtr((short *)(a));
  ByVal(*(short *)(a));
}

struct Inner { int m; };
struct Outer1 { struct Inner i; };
struct Outer2 { struct Outer1 o; };
struct Inner i;
struct Outer2 o;

void ByValInner (struct Inner);
void ByValOuter2 (struct Outer2);

void Inherit() {
  // Check we see through multiple levels
  int in;

  ByValOuter2(*(struct Outer2 *)&in);
  ByValOuter2(*(struct Outer2 *)&i);
  ByValInner(*(struct Inner *)&o.o);
  ByValInner(*(struct Inner *)&o);
  ByVal(*(int *)&o);
}

// PR 50066
typedef unsigned char uchar;

void Double(double);

int main() {
  double d = 2.34;
  int i[2];
  Double(d);

  // GCC: 1, 2, 3
  // level23-warning@+2{{type-punned pointer breaks}}
  // level1-warning@+1{{type-punned pointer might break}}
  *(long long *)i =
  // level123-note@-1{{not alias compatible}}

      // GCC: 1, 2, 3
      // level23-warning@+2{{type-punned pointer breaks}}
      // level1-warning@+1{{type-punned pointer might break}}
      *(long long *)&d;
      // level123-note@-1{{not alias compatible}}

  // GCC: 1, 2, 3
  // level23-warning@+2{{type-punned pointer breaks}}
  // level1-warning@+1{{type-punned pointer might break}}
  ((int *)&d)[0] = i[0];
  // level123-note@-1{{not alias compatible}}

  // GCC: 1, 2
  // level23-warning@+2{{type-punned pointer breaks}}
  // level1-warning@+1{{type-punned pointer might break}}
  ((int *)&d)[1] = i[1];
  // level123-note@-1{{not alias compatible}}

  Double(d);
  ((uchar *)&d)[2] += 123;
  Double(d);
  return 0;
}

// GCC gets (cast)[0], but not (cast)[1] because it represents the first as
// *(cast), and so it falls into the indirect operator path.

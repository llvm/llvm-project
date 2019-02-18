// Given code 'struct aa { char s1[4]; char * s2;} a; memcpy(a.s1, ...);',
// this test checks that the CStringChecker only invalidates the destination buffer array a.s1 (instead of a.s1 and a.s2).
// At the moment the whole of the destination array content is invalidated.
// If a.s1 region has a symbolic offset, the whole region of 'a' is invalidated.
// Specific triple set to test structures of size 0.
// RUN: %clang_analyze_cc1 -triple x86_64-pc-linux-gnu -analyzer-checker=core,unix.Malloc,debug.ExprInspection -analyzer-store=region -verify -analyzer-config eagerly-assume=false %s

typedef __typeof(sizeof(int)) size_t;

char *strdup(const char *s);
void free(void *);
void *memcpy(void *dst, const void *src, size_t n); // expected-note{{passing argument to parameter 'dst' here}}
void *malloc(size_t n);

void clang_analyzer_eval(int);

struct aa {
    char s1[4];
    char *s2;
};

// Test different types of structure initialisation.
int f0() {
  struct aa a0 = {{1, 2, 3, 4}, 0};
  a0.s2 = strdup("hello");
  char input[] = {'a', 'b', 'c', 'd'};
  memcpy(a0.s1, input, 4);
  clang_analyzer_eval(a0.s1[0] == 'a'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a0.s1[1] == 'b'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a0.s1[2] == 'c'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a0.s1[3] == 'd'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a0.s2 == 0); // expected-warning{{UNKNOWN}}
  free(a0.s2); // no warning
  return 0;
}

int f1() {
  struct aa a1;
  a1.s2 = strdup("hello");
  char input[] = {'a', 'b', 'c', 'd'};
  memcpy(a1.s1, input, 4);
  clang_analyzer_eval(a1.s1[0] == 'a'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a1.s1[1] == 'b'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a1.s1[2] == 'c'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a1.s1[3] == 'd'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a1.s2 == 0); // expected-warning{{UNKNOWN}}
  free(a1.s2); // no warning
  return 0;
}

int f2() {
  struct aa a2 = {{1, 2}};
  a2.s2 = strdup("hello");
  char input[] = {'a', 'b', 'c', 'd'};
  memcpy(a2.s1, input, 4);
  clang_analyzer_eval(a2.s1[0] == 'a'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a2.s1[1] == 'b'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a2.s1[2] == 'c'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a2.s1[3] == 'd'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a2.s2 == 0); // expected-warning{{UNKNOWN}}
  free(a2.s2); // no warning
  return 0;
}

int f3() {
  struct aa a3 = {{1, 2, 3, 4}, 0};
  a3.s2 = strdup("hello");
  char input[] = {'a', 'b', 'c', 'd'};
  int * dest = (int*)a3.s1;
  memcpy(dest, input, 4);
  clang_analyzer_eval(a3.s1[0] == 'a'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(dest[0] == 'a'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a3.s1[1] == 'b'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(dest[1] == 'b'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a3.s1[2] == 'c'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(dest[2] == 'c'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a3.s1[3] == 'd'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(dest[3] == 'd'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a3.s2 == 0); // expected-warning{{UNKNOWN}}
  free(a3.s2); // no warning
  return 0;
}

struct bb {
  struct aa a;
  char * s2;
};

int f4() {
  struct bb b0 = {{1, 2, 3, 4}, 0};
  b0.s2 = strdup("hello");
  b0.a.s2 = strdup("hola");
  char input[] = {'a', 'b', 'c', 'd'};
  char * dest = (char*)(b0.a.s1);
  memcpy(dest, input, 4);
  clang_analyzer_eval(b0.a.s1[0] == 'a'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(dest[0] == 'a'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(b0.a.s1[1] == 'b'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(dest[1] == 'b'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(b0.a.s1[2] == 'c'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(dest[2] == 'c'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(b0.a.s1[3] == 'd'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(dest[3] == 'd'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(b0.s2 == 0); // expected-warning{{UNKNOWN}}
  free(b0.a.s2); // no warning
  free(b0.s2); // no warning
  return 0;
}

// Test that memory leaks are caught.
int f5() {
  struct aa a0 = {{1, 2, 3, 4}, 0};
  a0.s2 = strdup("hello");
  char input[] = {'a', 'b', 'c', 'd'};
  memcpy(a0.s1, input, 4);
  return 0; // expected-warning{{Potential leak of memory pointed to by 'a0.s2'}}
}

int f6() {
  struct aa a1;
  a1.s2 = strdup("hello");
  char input[] = {'a', 'b', 'c', 'd'};
  memcpy(a1.s1, input, 4);
  return 0; // expected-warning{{Potential leak of memory pointed to by 'a1.s2'}}
}

int f7() {
  struct aa a2 = {{1, 2}};
  a2.s2 = strdup("hello");
  char input[] = {'a', 'b', 'c', 'd'};
  memcpy(a2.s1, input, 4);
  return 0; // expected-warning{{Potential leak of memory pointed to by 'a2.s2'}}
}

int f8() {
  struct aa a3 = {{1, 2, 3, 4}, 0};
  a3.s2 = strdup("hello");
  char input[] = {'a', 'b', 'c', 'd'};
  int * dest = (int*)a3.s1;
  memcpy(dest, input, 4);
  return 0; // expected-warning{{Potential leak of memory pointed to by 'a3.s2'}}
}

int f9() {
  struct bb b0 = {{1, 2, 3, 4}, 0};
  b0.s2 = strdup("hello");
  b0.a.s2 = strdup("hola");
  char input[] = {'a', 'b', 'c', 'd'};
  char * dest = (char*)(b0.a.s1);
  memcpy(dest, input, 4);
  free(b0.a.s2); // expected-warning{{Potential leak of memory pointed to by 'b0.s2'}}
  return 0;
}

int f10() {
  struct bb b0 = {{1, 2, 3, 4}, 0};
  b0.s2 = strdup("hello");
  b0.a.s2 = strdup("hola");
  char input[] = {'a', 'b', 'c', 'd'};
  char * dest = (char*)(b0.a.s1);
  memcpy(dest, input, 4);
  free(b0.s2); // expected-warning{{Potential leak of memory pointed to by 'b0.a.s2'}}
  return 0;
}

// Test invalidating fields being addresses of array.
struct cc {
  char * s1;
  char * s2;
};

int f11() {
  char x[4] = {1, 2};
  x[0] = 1;
  x[1] = 2;
  struct cc c0;
  c0.s2 = strdup("hello");
  c0.s1 = &x[0];
  char input[] = {'a', 'b', 'c', 'd'};
  memcpy(c0.s1, input, 4);
  clang_analyzer_eval(x[0] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(x[1] == 2); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(c0.s1[0] == 'a'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(c0.s1[1] == 'b'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(c0.s1[2] == 'c'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(c0.s1[3] == 'd'); // expected-warning{{UNKNOWN}}
  free(c0.s2); // no-warning
  return 0;
}

// Test inverting field position between s1 and s2.
struct dd {
  char *s2;
  char s1[4];
};

int f12() {
  struct dd d0 = {0, {1, 2, 3, 4}};
  d0.s2 = strdup("hello");
  char input[] = {'a', 'b', 'c', 'd'};
  memcpy(d0.s1, input, 4);
  clang_analyzer_eval(d0.s1[0] == 'a'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(d0.s1[1] == 'b'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(d0.s1[2] == 'c'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(d0.s1[3] == 'd'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(d0.s2 == 0); // expected-warning{{UNKNOWN}}
  free(d0.s2); // no warning
  return 0;
}

// Test arrays of structs.
struct ee {
  int a;
  char b;
};

struct EE {
  struct ee s1[2];
  char * s2;
};

int f13() {
  struct EE E0 = {{{1, 2}, {3, 4}}, 0};
  E0.s2 = strdup("hello");
  char input[] = {'a', 'b', 'c', 'd'};
  memcpy(E0.s1, input, 4);
  clang_analyzer_eval(E0.s1[0].a == 'a'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(E0.s1[0].b == 'b'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(E0.s1[1].a == 'c'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(E0.s1[1].b == 'd'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(E0.s2 == 0); // expected-warning{{UNKNOWN}}
  free(E0.s2); // no warning
  return 0;
}

// Test global parameters.
struct aa a15 = {{1, 2, 3, 4}, 0};

int f15() {
  a15.s2 = strdup("hello");
  char input[] = {'a', 'b', 'c', 'd'};
  memcpy(a15.s1, input, 4);
  clang_analyzer_eval(a15.s1[0] == 'a'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a15.s1[1] == 'b'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a15.s1[2] == 'c'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a15.s1[3] == 'd'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a15.s2 == 0); // expected-warning{{UNKNOWN}}
  free(a15.s2); // no warning
  return 0;
}

// Test array of 0 sized elements.
struct empty {};
struct gg {
  struct empty s1[4];
  char * s2;
};

int f16() {
  struct gg g0 = {{}, 0};
  g0.s2 = strdup("hello");
  char input[] = {'a', 'b', 'c', 'd'};
  memcpy(g0.s1, input, 4);
  clang_analyzer_eval(*(int*)(&g0.s1[0]) == 'a'); // expected-warning{{UNKNOWN}}\
  expected-warning{{Potential leak of memory pointed to by 'g0.s2'}}
  clang_analyzer_eval(*(int*)(&g0.s1[1]) == 'b'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(*(int*)(&g0.s1[2]) == 'c'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(*(int*)(&g0.s1[3]) == 'd'); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(g0.s2 == 0); // expected-warning{{UNKNOWN}}
  free(g0.s2); // no warning
  return 0;
}

// Test array of 0 elements.
struct hh {
  char s1[0];
  char * s2;
};

int f17() {
  struct hh h0;
  h0.s2 = strdup("hello");
  char input[] = {'a', 'b', 'c', 'd'};
  memcpy(h0.s1, input, 4);
  clang_analyzer_eval(h0.s1[0] == 'a'); // expected-warning{{UNKNOWN}}\
  expected-warning{{Potential leak of memory pointed to by 'h0.s2'}}
  clang_analyzer_eval(h0.s2 == 0); // expected-warning{{UNKNOWN}}
  free(h0.s2); // no warning
  return 0;
}

// Test writing past the array.
struct ii {
  char s1[4];
  int i;
  int j;
  char * s2;
};

int f18() {
  struct ii i18 = {{1, 2, 3, 4}, 5, 6};
  i18.i = 10;
  i18.j = 11;
  i18.s2 = strdup("hello");
  char input[100] = {3};
  memcpy(i18.s1, input, 100);
  clang_analyzer_eval(i18.s1[0] == 1); // expected-warning{{UNKNOWN}}\
  expected-warning{{Potential leak of memory pointed to by 'i18.s2'}}
  clang_analyzer_eval(i18.s1[1] == 2); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(i18.s1[2] == 3); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(i18.s1[3] == 4); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(i18.i == 10); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(i18.j == 11); // expected-warning{{UNKNOWN}}
  return 0;
}

int f181() {
  struct ii i181 = {{1, 2, 3, 4}, 5, 6};
  i181.i = 10;
  i181.j = 11;
  i181.s2 = strdup("hello");
  char input[100] = {3};
  memcpy(i181.s1, input, 5); // invalidate the whole region of i181
  clang_analyzer_eval(i181.s1[0] == 1); // expected-warning{{UNKNOWN}}\
  expected-warning{{Potential leak of memory pointed to by 'i181.s2'}}
  clang_analyzer_eval(i181.s1[1] == 2); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(i181.s1[2] == 3); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(i181.s1[3] == 4); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(i181.i == 10); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(i181.j == 11); // expected-warning{{UNKNOWN}}
  return 0;
}

// Test array with a symbolic offset.
struct jj {
  char s1[2];
  char * s2;
};

struct JJ {
  struct jj s1[3];
  char * s2;
};

int f19(int i) {
  struct JJ J0 = {{{1, 2, 0}, {3, 4, 0}, {5, 6, 0}}, 0};
  J0.s2 = strdup("hello");
  J0.s1[0].s2 = strdup("hello");
  J0.s1[1].s2 = strdup("hi");
  J0.s1[2].s2 = strdup("world");
  char input[2] = {'a', 'b'};
  memcpy(J0.s1[i].s1, input, 2);
  clang_analyzer_eval(J0.s1[0].s1[0] == 1); // expected-warning{{UNKNOWN}}\
  expected-warning{{Potential leak of memory pointed to by field 's2'}}\
  expected-warning{{Potential leak of memory pointed to by 'J0.s2'}}
  clang_analyzer_eval(J0.s1[0].s1[1] == 2); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(J0.s1[1].s1[0] == 3); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(J0.s1[1].s1[1] == 4); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(J0.s1[2].s1[0] == 5); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(J0.s1[2].s1[1] == 6); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(J0.s1[i].s1[0] == 5); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(J0.s1[i].s1[1] == 6); // expected-warning{{UNKNOWN}}
  // FIXME: memory leak warning for J0.s2 should be emitted here instead of after memcpy call.
  return 0; // no warning
}

// Test array with its super region having symbolic offseted regions.
int f20(int i) {
  struct aa * a20 = malloc(sizeof(struct aa) * 2);
  a20[0].s1[0] = 1;
  a20[0].s1[1] = 2;
  a20[0].s1[2] = 3;
  a20[0].s1[3] = 4;
  a20[0].s2 = strdup("hello");
  a20[1].s1[0] = 5;
  a20[1].s1[1] = 6;
  a20[1].s1[2] = 7;
  a20[1].s1[3] = 8;
  a20[1].s2 = strdup("world");
  a20[i].s2 = strdup("hola");
  char input[] = {'a', 'b', 'c', 'd'};
  memcpy(a20[0].s1, input, 4);
  clang_analyzer_eval(a20[0].s1[0] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a20[0].s1[1] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a20[0].s1[2] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a20[0].s1[3] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a20[0].s2 == 0); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a20[1].s1[0] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a20[1].s1[1] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a20[1].s1[2] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a20[1].s1[3] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a20[1].s2 == 0); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a20[i].s1[0] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a20[i].s1[1] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a20[i].s1[2] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a20[i].s1[3] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a20[i].s2 == 0); // expected-warning{{UNKNOWN}}\
  expected-warning{{Potential leak of memory pointed to by 'a20'}}

  return 0;
}

// Test array's region and super region both having symbolic offsets.
int f21(int i) {
  struct aa * a21 = malloc(sizeof(struct aa) * 2);
  a21[0].s1[0] = 1;
  a21[0].s1[1] = 2;
  a21[0].s1[2] = 3;
  a21[0].s1[3] = 4;
  a21[0].s2 = 0;
  a21[1].s1[0] = 5;
  a21[1].s1[1] = 6;
  a21[1].s1[2] = 7;
  a21[1].s1[3] = 8;
  a21[1].s2 = 0;
  a21[i].s2 = strdup("hello");
  a21[i].s1[0] = 1;
  a21[i].s1[1] = 2;
  a21[i].s1[2] = 3;
  a21[i].s1[3] = 4;
  char input[] = {'a', 'b', 'c', 'd'};
  memcpy(a21[i].s1, input, 4);
  clang_analyzer_eval(a21[0].s1[0] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a21[0].s1[1] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a21[0].s1[2] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a21[0].s1[3] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a21[0].s2 == 0); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a21[1].s1[0] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a21[1].s1[1] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a21[1].s1[2] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a21[1].s1[3] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a21[1].s2 == 0); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a21[i].s1[0] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a21[i].s1[1] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a21[i].s1[2] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a21[i].s1[3] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a21[i].s2 == 0); // expected-warning{{UNKNOWN}}\
  expected-warning{{Potential leak of memory pointed to by 'a21'}}

  return 0;
}

// Test regions aliasing other regions.
struct ll {
  char s1[4];
  char * s2;
};

struct mm {
  char s3[4];
  char * s4;
};

int f24() {
  struct ll l24 = {{1, 2, 3, 4}, 0};
  struct mm * m24 = (struct mm *)&l24;
  m24->s4 = strdup("hello");
  char input[] = {1, 2, 3, 4};
  memcpy(m24->s3, input, 4);
  clang_analyzer_eval(m24->s3[0] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(m24->s3[1] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(m24->s3[2] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(m24->s3[3] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(l24.s1[0] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(l24.s1[1] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(l24.s1[2] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(l24.s1[3] == 1); // expected-warning{{UNKNOWN}}\
  expected-warning{{Potential leak of memory pointed to by field 's4'}}
  return 0;
}

// Test region with potential aliasing and symbolic offsets.
// Store assumes no aliasing.
int f25(int i, int j, struct ll * l, struct mm * m) {
  m->s4 = strdup("hola"); // m->s4 not tracked
  m->s3[0] = 1;
  m->s3[1] = 2;
  m->s3[2] = 3;
  m->s3[3] = 4;
  m->s3[j] = 5; // invalidates m->s3
  l->s2 = strdup("hello"); // l->s2 not tracked
  l->s1[0] = 6;
  l->s1[1] = 7;
  l->s1[2] = 8;
  l->s1[3] = 9;
  l->s1[i] = 10; // invalidates l->s1
  char input[] = {1, 2, 3, 4};
  memcpy(m->s3, input, 4); // does not invalidate l->s1[i]
  clang_analyzer_eval(m->s3[0] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(m->s3[1] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(m->s3[2] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(m->s3[3] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(m->s3[i] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(m->s3[j] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(l->s1[0] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(l->s1[1] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(l->s1[2] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(l->s1[3] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(l->s1[i] == 1); // expected-warning{{FALSE}}
  clang_analyzer_eval(l->s1[j] == 1); // expected-warning{{UNKNOWN}}
  return 0;
}

// Test size with symbolic size argument.
int f26(int i) {
  struct aa a26 = {{1, 2, 3, 4}, 0};
  a26.s2 = strdup("hello");
  char input[] = {'a', 'b', 'c', 'd'};
  memcpy(a26.s1, input, i); // i assumed in bound
  clang_analyzer_eval(a26.s1[0] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a26.s1[1] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a26.s1[2] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a26.s1[3] == 1); // expected-warning{{UNKNOWN}}\
  expected-warning{{Potential leak of memory pointed to by 'a26.s2'}}
  return 0;
}

// Test sizeof as a size argument.
int f261() {
  struct aa a261 = {{1, 2, 3, 4}, 0};
  a261.s2 = strdup("hello");
  char input[] = {'a', 'b', 'c', 'd'};
  memcpy(a261.s1, input, sizeof(a261.s1));
  clang_analyzer_eval(a261.s1[0] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a261.s1[1] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a261.s1[2] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a261.s1[3] == 1); // expected-warning{{UNKNOWN}}\
  expected-warning{{Potential leak of memory pointed to by 'a261.s2'}}
  return 0;
}

// Test negative size argument.
int f262() {
  struct aa a262 = {{1, 2, 3, 4}, 0};
  a262.s2 = strdup("hello");
  char input[] = {'a', 'b', 'c', 'd'};
  memcpy(a262.s1, input, -1);
  clang_analyzer_eval(a262.s1[0] == 1); // expected-warning{{UNKNOWN}}\
  expected-warning{{Potential leak of memory pointed to by 'a262.s2'}}
  clang_analyzer_eval(a262.s1[1] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a262.s1[2] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a262.s1[3] == 1); // expected-warning{{UNKNOWN}}
  return 0;
}

// Test size argument being an unknown value.
struct xx {
  char s1[4];
  char * s2;
};

int f263(int n, char * len) {
  struct xx x263 = {0};
  x263.s2 = strdup("hello");
  char input[] = {'a', 'b', 'c', 'd'};
  memcpy(x263.s1, input, *(len + n));
  clang_analyzer_eval(x263.s1[0] == 0); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(x263.s1[1] == 0); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(x263.s1[2] == 0); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(x263.s1[3] == 0); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(x263.s2 == 0); // expected-warning{{UNKNOWN}}
  return 0; // expected-warning{{Potential leak of memory pointed to by 'x263.s2'}}
}


// Test casting regions with symbolic offseted sub regions.
int f27(int i) {
  struct mm m27 = {{1, 2, 3, 4}, 0};
  m27.s4 = strdup("hello");
  m27.s3[i] = 5;
  char input[] = {'a', 'b', 'c', 'd'};
  memcpy(((struct ll*)(&m27))->s1, input, 4);
  clang_analyzer_eval(m27.s3[0] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(m27.s3[1] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(m27.s3[2] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(m27.s3[3] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(m27.s3[i] == 1); // expected-warning{{UNKNOWN}}\
  expected-warning{{Potential leak of memory pointed to by 'm27.s4'}}
  return 0;
}

int f28(int i, int j, int k, int l) {
  struct mm m28[2];
  m28[i].s4 = strdup("hello");
  m28[j].s3[k] = 1;
  struct ll * l28 = (struct ll*)(&m28[1]);
  l28->s1[l] = 2;
  char input[] = {'a', 'b', 'c', 'd'}; // expected-warning{{Potential leak of memory pointed to by field 's4'}}
  memcpy(l28->s1, input, 4);
  clang_analyzer_eval(m28[0].s3[0] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(m28[0].s3[1] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(m28[0].s3[2] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(m28[0].s3[3] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(m28[1].s3[0] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(m28[1].s3[1] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(m28[1].s3[2] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(m28[1].s3[3] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(m28[i].s3[0] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(m28[i].s3[1] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(m28[i].s3[2] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(m28[i].s3[3] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(m28[j].s3[k] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(l28->s1[l] == 2); // expected-warning{{UNKNOWN}}
  return 0;
}

int f29(int i, int j, int k, int l, int m) {
  struct mm m29[2];
  m29[i].s4 = strdup("hello");
  m29[j].s3[k] = 1;
  struct ll * l29 = (struct ll*)(&m29[l]);
  l29->s1[m] = 2;
  char input[] = {'a', 'b', 'c', 'd'};
  memcpy(l29->s1, input, 4);
  clang_analyzer_eval(m29[0].s3[0] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(m29[0].s3[1] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(m29[0].s3[2] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(m29[0].s3[3] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(m29[1].s3[0] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(m29[1].s3[1] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(m29[1].s3[2] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(m29[1].s3[3] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(m29[i].s3[0] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(m29[i].s3[1] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(m29[i].s3[2] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(m29[i].s3[3] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(m29[j].s3[k] == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(l29->s1[m] == 2); // expected-warning{{UNKNOWN}}
  // FIXME: Should warn that m29[i].s4 leaks. But not on the previous line,
  // because l29 and m29 alias.
  return 0;
}

// Test unions' fields.
union uu {
  char x;
  char s1[4];
};

int f30() {
  union uu u30 = { .s1 = {1, 2, 3, 4}};
  char input[] = {1, 2, 3, 4};
  memcpy(u30.s1, input, 4);
  clang_analyzer_eval(u30.s1[0] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(u30.s1[1] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(u30.s1[2] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(u30.s1[3] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(u30.x == 1); // expected-warning{{UNKNOWN}}
  return 0;
}

struct kk {
  union uu u;
  char * s2;
};

int f31() {
  struct kk k31;
  k31.s2 = strdup("hello");
  k31.u.x = 1;
  char input[] = {'a', 'b', 'c', 'd'};
  memcpy(k31.u.s1, input, 4);
  clang_analyzer_eval(k31.u.s1[0] == 1); // expected-warning{{UNKNOWN}}\
  expected-warning{{Potential leak of memory pointed to by 'k31.s2'}}
  clang_analyzer_eval(k31.u.s1[1] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(k31.u.s1[2] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(k31.u.s1[3] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(k31.u.x == 1); // expected-warning{{UNKNOWN}}
  // FIXME: memory leak warning for k31.s2 should be emitted here.
  return 0;
}

union vv {
  int x;
  char * s2;
};

int f32() {
  union vv v32;
  v32.s2 = strdup("hello");
  char input[] = {'a', 'b', 'c', 'd'};
  memcpy(v32.s2, input, 4);
  clang_analyzer_eval(v32.s2[0] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(v32.s2[1] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(v32.s2[2] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(v32.s2[3] == 1); // expected-warning{{UNKNOWN}}\
  expected-warning{{Potential leak of memory pointed to by 'v32.s2'}}
  return 0;
}

struct nn {
  int s1;
  int i;
  int j;
  int k;
  char * s2;
};

// Test bad types to dest buffer.
int f33() {
  struct nn n33 = {1, 2, 3, 4, 0};
  n33.s2 = strdup("hello");
  char input[] = {'a', 'b', 'c', 'd'};
  memcpy(n33.s1, input, 4); // expected-warning{{incompatible integer to pointer conversion passing 'int' to parameter of type 'void *'}}
  clang_analyzer_eval(n33.i == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(n33.j == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(n33.k == 4); // expected-warning{{TRUE}}
  clang_analyzer_eval(((char*)(n33.s1))[0] == 1); // expected-warning{{UNKNOWN}}\
  expected-warning{{cast to 'char *' from smaller integer type 'int'}}
  clang_analyzer_eval(((char*)(n33.s1))[1] == 1); // expected-warning{{UNKNOWN}}\
  expected-warning{{cast to 'char *' from smaller integer type 'int'}}
  clang_analyzer_eval(((char*)(n33.s1))[2] == 1); // expected-warning{{UNKNOWN}}\
  expected-warning{{cast to 'char *' from smaller integer type 'int'}}
  clang_analyzer_eval(((char*)(n33.s1))[3] == 1); // expected-warning{{UNKNOWN}}\
  expected-warning{{cast to 'char *' from smaller integer type 'int'}}
  clang_analyzer_eval(n33.s2 == 0); //expected-warning{{UNKNOWN}}
  return 0; // expected-warning{{Potential leak of memory pointed to by 'n33.s2'}}
}

// Test destination buffer being an unknown value.
struct ww {
  int s1[4];
  char s2;
};

int f34(struct ww * w34, int n) {
  w34->s2 = 3;
  char input[] = {'a', 'b', 'c', 'd'};
  memcpy(w34->s1 + n, input , 4);
  clang_analyzer_eval(w34->s1[0] == 0); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(w34->s1[1] == 0); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(w34->s1[2] == 0); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(w34->s1[3] == 0); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(w34->s1[n] == 0); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(w34->s2 == 3); // expected-warning{{TRUE}}
  return 0;
}

// Test dest buffer as an element region with a symbolic index and size parameter as a symbolic value.
struct yy {
  char s1[4];
  char * s2;
};

int f35(int i, int n) {
  struct yy y35 = {{1, 2, 3, 4}, 0};
  y35.s2 = strdup("hello");
  char input[] = {'a', 'b', 'c', 'd'};
  memcpy(&(y35.s1[i]), input, n);
  clang_analyzer_eval(y35.s1[0] == 0); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(y35.s1[1] == 0); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(y35.s1[2] == 0); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(y35.s1[3] == 0); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(y35.s1[i] == 0); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(y35.s2 == 0); // expected-warning{{UNKNOWN}}
  return 0; // expected-warning{{Potential leak of memory pointed to by 'y35.s2'}}
}

// Test regions with negative offsets.
struct zz {
  char s1[4];
  int s2;
};

int f36(struct zz * z36) {

  char input[] = {'a', 'b', 'c', 'd'};
  z36->s1[0] = 0;
  z36->s1[1] = 1;
  z36->s1[2] = 2;
  z36->s1[3] = 3;
  z36->s2 = 10;

  z36 = z36 - 1; // Decrement by 8 bytes (struct zz is 8 bytes).

  z36->s1[0] = 4;
  z36->s1[1] = 5;
  z36->s1[2] = 6;
  z36->s1[3] = 7;
  z36->s2 = 11;

  memcpy(z36->s1, input, 4);

  clang_analyzer_eval(z36->s1[0] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(z36->s1[1] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(z36->s1[2] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(z36->s1[3] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(z36->s2 == 11); // expected-warning{{TRUE}}

  z36 = z36 + 1; // Increment back.

  clang_analyzer_eval(z36->s1[0] == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(z36->s1[1] == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(z36->s1[2] == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(z36->s1[3] == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(z36->s2 == 10); // expected-warning{{TRUE}}

  return 0;
}

int f37(struct zz * z37) {

  char input[] = {'a', 'b', 'c', 'd'};
  z37->s1[0] = 0;
  z37->s1[1] = 1;
  z37->s1[2] = 2;
  z37->s1[3] = 3;
  z37->s2 = 10;

  z37 = (struct zz *)((char*)(z37) - 4); // Decrement by 4 bytes (struct zz is 8 bytes).

  z37->s1[0] = 4;
  z37->s1[1] = 5;
  z37->s1[2] = 6;
  z37->s1[3] = 7;
  z37->s2 = 11;

  memcpy(z37->s1, input, 4);

  clang_analyzer_eval(z37->s1[0] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(z37->s1[1] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(z37->s1[2] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(z37->s1[3] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(z37->s2 == 11); // expected-warning{{TRUE}}

  z37 = (struct zz *)((char*)(z37) + 4); // Increment back.

  clang_analyzer_eval(z37->s1[0] == 11); // expected-warning{{TRUE}}
  clang_analyzer_eval(z37->s1[1] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(z37->s1[2] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(z37->s1[3] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(z37->s2 == 10); // expected-warning{{TRUE}}

  return 0;
}

int f38(struct zz * z38) {

  char input[] = {'a', 'b', 'c', 'd'};
  z38->s1[0] = 0;
  z38->s1[1] = 1;
  z38->s1[2] = 2;
  z38->s1[3] = 3;
  z38->s2 = 10;

  z38 = (struct zz *)((char*)(z38) - 2); // Decrement by 2 bytes (struct zz is 8 bytes).

  z38->s1[0] = 4;
  z38->s1[1] = 5;
  z38->s1[2] = 6;
  z38->s1[3] = 7;
  z38->s2 = 11;

  memcpy(z38->s1, input, 4);

  clang_analyzer_eval(z38->s1[0] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(z38->s1[1] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(z38->s1[2] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(z38->s1[3] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(z38->s2 == 11); // expected-warning{{TRUE}}

  z38 = (struct zz *)((char*)(z38) + 2); // Increment back.

  clang_analyzer_eval(z38->s1[0] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(z38->s1[1] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(z38->s1[2] == 11); // expected-warning{{TRUE}}
  clang_analyzer_eval(z38->s1[3] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(z38->s2 == 10); // expected-warning{{UNKNOWN}}

  return 0;
}

// Test negative offsets with a different structure layout.
struct z0 {
  int s2;
  char s1[4];
};

int f39(struct z0 * d39) {

  char input[] = {'a', 'b', 'c', 'd'};
  d39->s1[0] = 0;
  d39->s1[1] = 1;
  d39->s1[2] = 2;
  d39->s1[3] = 3;
  d39->s2 = 10;

  d39 = (struct z0 *)((char*)(d39) - 2); // Decrement by 2 bytes (struct z0 is 8 bytes).

  d39->s1[0] = 4;
  d39->s1[1] = 5;
  d39->s1[2] = 6;
  d39->s1[3] = 7;
  d39->s2 = 11;

  memcpy(d39->s1, input, 4);

  clang_analyzer_eval(d39->s1[0] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(d39->s1[1] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(d39->s1[2] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(d39->s1[3] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(d39->s2 == 11); // expected-warning{{TRUE}}

  d39 = (struct z0 *)((char*)(d39) + 2); // Increment back.

  clang_analyzer_eval(d39->s1[0] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(d39->s1[1] == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(d39->s1[2] == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(d39->s1[3] == 3); // expected-warning{{TRUE}}
  // FIXME: d39->s2 should evaluate to at least UNKNOWN or FALSE,
  // 'collectSubRegionBindings(...)' in RegionStore.cpp will need to
  // handle a regions' upper boundary overflowing.
  clang_analyzer_eval(d39->s2 == 10); // expected-warning{{TRUE}}

  return 0;
}


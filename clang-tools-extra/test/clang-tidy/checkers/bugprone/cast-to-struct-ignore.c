// RUN: %check_clang_tidy %s bugprone-cast-to-struct %t -- \
// RUN:   -config="{CheckOptions: {bugprone-cast-to-struct.IgnoredCasts: 'int \*;struct S1 \*;TYPE \*;struct S2 \*;TYPE_P;struct S3 \*;struct S4 \*;struct S. \*'}}"

struct S1 {
};

struct S2 {
};

struct S3 {
};

struct S4 {
};

typedef int TYPE;
typedef int * TYPE_P;
typedef unsigned char uchar;

void test1(int *p0, TYPE *p1, TYPE_P p2) {
  struct S1 *s1;
  struct S4 *s4;

  s1 = (struct S1 *)p0; // no warning
  s1 = (struct S1 *)p1; // CHECK-MESSAGES: warning: casting a 'TYPE *' (aka 'int *') pointer to a 'struct S1 *'
  s1 = (struct S1 *)p2; // CHECK-MESSAGES: warning: casting a 'TYPE_P' (aka 'int *') pointer to a 'struct S1 *'
  s4 = (struct S4 *)p0; // CHECK-MESSAGES: warning: casting a 'int *' pointer to a 'struct S4 *'
}

void test2(int *p0, TYPE *p1, TYPE_P p2) {
  struct S2 *s2;
  struct S4 *s4;

  s2 = (struct S2 *)p0; // CHECK-MESSAGES: warning: casting a 'int *' pointer to a 'struct S2 *'
  s2 = (struct S2 *)p1; // no warning
  s2 = (struct S2 *)p2; // CHECK-MESSAGES: warning: casting a 'TYPE_P' (aka 'int *') pointer to a 'struct S2 *'
  s4 = (struct S4 *)p1; // CHECK-MESSAGES: warning: casting a 'TYPE *' (aka 'int *') pointer to a 'struct S4 *'
}

void test3(int *p0, TYPE *p1, TYPE_P p2) {
  struct S3 *s3;
  struct S4 *s4;

  s3 = (struct S3 *)p0; // CHECK-MESSAGES: warning: casting a 'int *' pointer to a 'struct S3 *'
  s3 = (struct S3 *)p1; // CHECK-MESSAGES: warning: casting a 'TYPE *' (aka 'int *') pointer to a 'struct S3 *'
  s3 = (struct S3 *)p2; // no warning
  s4 = (struct S4 *)p2; // CHECK-MESSAGES: warning: casting a 'TYPE_P' (aka 'int *') pointer to a 'struct S4 *'
}

void test_wildcard(struct S4 *p1, struct S1 *p2) {
  struct S1 *s1;
  struct S2 *s2;
  s1 = (struct S1 *)p1;
  s2 = (struct S2 *)p1;
  s2 = (struct S2 *)p2; // CHECK-MESSAGES: warning: casting a 'struct S1 *' pointer to a 'struct S2 *'
}

void test_default_ignore(void *p1, char *p2, uchar *p3) {
  struct S4 *s4;
  s4 = (struct S4 *)p1;
  s4 = (struct S4 *)p2;
  s4 = (struct S4 *)p3;
}

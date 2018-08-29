// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.core,debug.ExprInspection -verify -analyzer-config eagerly-assume=false %s

void clang_analyzer_eval(int);

int string_literal_init() {
  char a[] = "abc";
  char b[2] = "abc"; // expected-warning{{too long}}
  char c[5] = "abc";

  clang_analyzer_eval(a[1] == 'b'); // expected-warning{{TRUE}}
  clang_analyzer_eval(b[1] == 'b'); // expected-warning{{TRUE}}
  clang_analyzer_eval(c[1] == 'b'); // expected-warning{{TRUE}}

  clang_analyzer_eval(a[3] == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(c[3] == 0); // expected-warning{{TRUE}}

  clang_analyzer_eval(c[4] == 0); // expected-warning{{TRUE}}

  return 42;
}

void nested_compound_literals(int rad) {
  int vec[6][2] = {{0.195, 0.02}, {0.383, 0.067}, {0.55, 0.169},  // expected-warning 6 {{implicit conversion from 'double' to 'int' changes value from}}
                   {0.831, 0.45}, {0.924, 0.617}, {0.98, 0.805}}; // expected-warning 6 {{implicit conversion from 'double' to 'int' changes value from}}
  int a;

  for (a = 0; a < 6; ++a) {
      vec[a][0] *= rad; // no-warning
      vec[a][1] *= rad; // no-warning
  }
}

void nested_compound_literals_float(float rad) {
  float vec[6][2] = {{0.195, 0.02}, {0.383, 0.067}, {0.55, 0.169},
                     {0.831, 0.45}, {0.924, 0.617}, {0.98, 0.805}};
  int a;

  for (a = 0; a < 6; ++a) {
      vec[a][0] *= rad; // no-warning
      vec[a][1] *= rad; // no-warning
  }
}


void struct_as_array() {
  struct simple { int x; int y; };
  struct simple a;
  struct simple *p = &a;

  p->x = 5;
  clang_analyzer_eval(a.x == 5); // expected-warning{{TRUE}}
  clang_analyzer_eval(p[0].x == 5); // expected-warning{{TRUE}}

  p[0].y = 5;
  clang_analyzer_eval(a.y == 5); // expected-warning{{TRUE}}
  clang_analyzer_eval(p->y == 5); // expected-warning{{TRUE}}
}


// PR13264 / <rdar://problem/11802440>
struct point { int x; int y; };
struct circle { struct point o; int r; };
struct circle get_circle() {
  struct circle result;
  result.r = 5;
  result.o = (struct point){0, 0};
  return result;
}

void struct_in_struct() {
  struct circle c;
  c = get_circle();
  // This used to think c.r was undefined because c.o is a LazyCompoundVal.
  clang_analyzer_eval(c.r == 5); // expected-warning{{TRUE}}
}

// We also test with floats because we don't model floats right now,
// and the original bug report used a float.
struct circle_f { struct point o; float r; };
struct circle_f get_circle_f() {
  struct circle_f result;
  result.r = 5.0;
  result.o = (struct point){0, 0};
  return result;
}

float struct_in_struct_f() {
  struct circle_f c;
  c = get_circle_f();

  return c.r; // no-warning
}


int randomInt();

int testSymbolicInvalidation(int index) {
  int vals[10];

  vals[0] = 42;
  clang_analyzer_eval(vals[0] == 42); // expected-warning{{TRUE}}

  vals[index] = randomInt();
  clang_analyzer_eval(vals[0] == 42); // expected-warning{{UNKNOWN}}

  return vals[index]; // no-warning
}

int testConcreteInvalidation(int index) {
  int vals[10];

  vals[index] = 42;
  clang_analyzer_eval(vals[index] == 42); // expected-warning{{TRUE}}
  vals[0] = randomInt();
  clang_analyzer_eval(vals[index] == 42); // expected-warning{{UNKNOWN}}

  return vals[0]; // no-warning
}


typedef struct {
  int x, y, z;
} S;

S makeS();

int testSymbolicInvalidationStruct(int index) {
  S vals[10];

  vals[0].x = 42;
  clang_analyzer_eval(vals[0].x == 42); // expected-warning{{TRUE}}

  vals[index] = makeS();
  clang_analyzer_eval(vals[0].x == 42); // expected-warning{{UNKNOWN}}

  return vals[index].x; // no-warning
}

int testConcreteInvalidationStruct(int index) {
  S vals[10];

  vals[index].x = 42;
  clang_analyzer_eval(vals[index].x == 42); // expected-warning{{TRUE}}
  vals[0] = makeS();
  clang_analyzer_eval(vals[index].x == 42); // expected-warning{{UNKNOWN}}

  return vals[0].x; // no-warning
}

typedef struct {
  S a[5];
  S b[5];
} SS;

int testSymbolicInvalidationDoubleStruct(int index) {
  SS vals;

  vals.a[0].x = 42;
  vals.b[0].x = 42;
  clang_analyzer_eval(vals.a[0].x == 42); // expected-warning{{TRUE}}
  clang_analyzer_eval(vals.b[0].x == 42); // expected-warning{{TRUE}}

  vals.a[index] = makeS();
  clang_analyzer_eval(vals.a[0].x == 42); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(vals.b[0].x == 42); // expected-warning{{TRUE}}

  return vals.b[index].x; // no-warning
}

int testConcreteInvalidationDoubleStruct(int index) {
  SS vals;

  vals.a[index].x = 42;
  vals.b[index].x = 42;
  clang_analyzer_eval(vals.a[index].x == 42); // expected-warning{{TRUE}}
  clang_analyzer_eval(vals.b[index].x == 42); // expected-warning{{TRUE}}

  vals.a[0] = makeS();
  clang_analyzer_eval(vals.a[index].x == 42); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(vals.b[index].x == 42); // expected-warning{{TRUE}}

  return vals.b[0].x; // no-warning
}


int testNonOverlappingStructFieldsSimple() {
  S val;

  val.x = 1;
  val.y = 2;
  clang_analyzer_eval(val.x == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(val.y == 2); // expected-warning{{TRUE}}

  return val.z; // expected-warning{{garbage}}
}

int testNonOverlappingStructFieldsSymbolicBase(int index, int anotherIndex) {
  SS vals;

  vals.a[index].x = 42;
  vals.a[index].y = 42;
  clang_analyzer_eval(vals.a[index].x == 42); // expected-warning{{TRUE}}
  clang_analyzer_eval(vals.a[index].y == 42); // expected-warning{{TRUE}}

  vals.a[anotherIndex].x = 42;
  clang_analyzer_eval(vals.a[index].x == 42); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(vals.a[index].y == 42); // expected-warning{{TRUE}}

  // FIXME: False negative. No bind ever set a field 'z'.
  return vals.a[index].z; // no-warning
}

int testStructFieldChains(int index, int anotherIndex) {
  SS vals[4];

  vals[index].a[0].x = 42;
  vals[anotherIndex].a[1].y = 42;
  clang_analyzer_eval(vals[index].a[0].x == 42); // expected-warning{{TRUE}}
  clang_analyzer_eval(vals[anotherIndex].a[1].y == 42); // expected-warning{{TRUE}}

  // This doesn't affect anything in the 'a' array field.
  vals[anotherIndex].b[1].x = 42;
  clang_analyzer_eval(vals[index].a[0].x == 42); // expected-warning{{TRUE}}
  clang_analyzer_eval(vals[anotherIndex].a[1].y == 42); // expected-warning{{TRUE}}
  clang_analyzer_eval(vals[anotherIndex].b[1].x == 42); // expected-warning{{TRUE}}

  // This doesn't affect anything in the 'b' array field.
  vals[index].a[anotherIndex].x = 42;
  clang_analyzer_eval(vals[index].a[0].x == 42); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(vals[anotherIndex].a[0].x == 42); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(vals[anotherIndex].a[1].y == 42); // expected-warning{{TRUE}}
  clang_analyzer_eval(vals[anotherIndex].b[1].x == 42); // expected-warning{{TRUE}}

  // FIXME: False negative. No bind ever set a field 'z'.
  return vals[index].a[0].z; // no-warning
}

int testStructFieldChainsNested(int index, int anotherIndex) {
  SS vals[4];

  vals[index].a[0].x = 42;
  clang_analyzer_eval(vals[index].a[0].x == 42); // expected-warning{{TRUE}}

  vals[index].b[0] = makeS();
  clang_analyzer_eval(vals[index].a[0].x == 42); // expected-warning{{TRUE}}

  vals[index].a[0] = makeS();
  clang_analyzer_eval(vals[index].a[0].x == 42); // expected-warning{{UNKNOWN}}

  vals[index].a[0].x = 42;
  clang_analyzer_eval(vals[index].a[0].x == 42); // expected-warning{{TRUE}}

  return 0;
}

typedef struct {
  int zoomLevel;
  struct point center;
} Outer;

extern int test13116945(struct point x);
static void radar13116945(struct point centerCoordinate) {
  Outer zoomRegion;
  zoomRegion.zoomLevel = 0;
  zoomRegion.center = centerCoordinate;
  Outer r = zoomRegion;
  test13116945(r.center); // no-warning
}


typedef struct {
  char data[4];
} ShortString;

typedef struct {
  ShortString str;
  int length;
} ShortStringWrapper;

void testArrayStructCopy() {
  ShortString s = { "abc" };
  ShortString s2 = s;
  ShortString s3 = s2;

  clang_analyzer_eval(s3.data[0] == 'a'); // expected-warning{{TRUE}}
  clang_analyzer_eval(s3.data[1] == 'b'); // expected-warning{{TRUE}}
  clang_analyzer_eval(s3.data[2] == 'c'); // expected-warning{{TRUE}}

  s3.data[0] = 'z';
  ShortString s4 = s3;

  clang_analyzer_eval(s4.data[0] == 'z'); // expected-warning{{TRUE}}
  clang_analyzer_eval(s4.data[1] == 'b'); // expected-warning{{TRUE}}
  clang_analyzer_eval(s4.data[2] == 'c'); // expected-warning{{TRUE}}
}

void testArrayStructCopyNested() {
  ShortString s = { "abc" };
  ShortString s2 = s;

  ShortStringWrapper w = { s2, 0 };

  clang_analyzer_eval(w.str.data[0] == 'a'); // expected-warning{{TRUE}}
  clang_analyzer_eval(w.str.data[1] == 'b'); // expected-warning{{TRUE}}
  clang_analyzer_eval(w.str.data[2] == 'c'); // expected-warning{{TRUE}}
  clang_analyzer_eval(w.length == 0); // expected-warning{{TRUE}}

  ShortStringWrapper w2 = w;
  clang_analyzer_eval(w2.str.data[0] == 'a'); // expected-warning{{TRUE}}
  clang_analyzer_eval(w2.str.data[1] == 'b'); // expected-warning{{TRUE}}
  clang_analyzer_eval(w2.str.data[2] == 'c'); // expected-warning{{TRUE}}
  clang_analyzer_eval(w2.length == 0); // expected-warning{{TRUE}}

  ShortStringWrapper w3 = w2;
  clang_analyzer_eval(w3.str.data[0] == 'a'); // expected-warning{{TRUE}}
  clang_analyzer_eval(w3.str.data[1] == 'b'); // expected-warning{{TRUE}}
  clang_analyzer_eval(w3.str.data[2] == 'c'); // expected-warning{{TRUE}}
  clang_analyzer_eval(w3.length == 0); // expected-warning{{TRUE}}
}

// --------------------
// False positives
// --------------------

int testMixSymbolicAndConcrete(int index, int anotherIndex) {
  SS vals;

  vals.a[index].x = 42;
  vals.a[0].y = 42;

  // FIXME: Should be TRUE.
  clang_analyzer_eval(vals.a[index].x == 42); // expected-warning{{UNKNOWN}}
  // Should be TRUE; we set this explicitly.
  clang_analyzer_eval(vals.a[0].y == 42); // expected-warning{{TRUE}}

  vals.a[anotherIndex].y = 42;

  // Should be UNKNOWN; we set an 'x'.
  clang_analyzer_eval(vals.a[index].x == 42); // expected-warning{{UNKNOWN}}
  // FIXME: Should be TRUE.
  clang_analyzer_eval(vals.a[0].y == 42); // expected-warning{{UNKNOWN}}

  return vals.a[0].x; // no-warning
}

void testFieldChainIsNotEnough(int index) {
  SS vals[4];

  vals[index].a[0].x = 42;
  clang_analyzer_eval(vals[index].a[0].x == 42); // expected-warning{{TRUE}}

  vals[index].a[1] = makeS();
  // FIXME: Should be TRUE.
  clang_analyzer_eval(vals[index].a[0].x == 42); // expected-warning{{UNKNOWN}}
}

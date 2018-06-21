// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-config suppress-inlined-defensive-checks=true -verify %s

// Perform inline defensive checks.
void idc(void *p) {
	if (p)
		;
}

int test01(int *p) {
  if (p)
    ;
  return *p; // expected-warning {{Dereference of null pointer}}
}

int test02(int *p, int *x) {
  if (p)
    ;
  idc(p);
	if (x)
		;
  return *p; // expected-warning {{Dereference of null pointer}}
}

int test03(int *p, int *x) {
	idc(p);
	if (p)
		;
	return *p; // False negative
}

int deref04(int *p) {
  return *p; // expected-warning {{Dereference of null pointer}}
}

int test04(int *p) {
  if (p)
    ;
  idc(p);
  return deref04(p);
}

int test11(int *q, int *x) {
	int *p = q;
	if (q)
		;
	if (x)
		;
	return *p; // expected-warning{{Dereference of null pointer}}
}

int test12(int *q) {
	int *p = q;
	idc(q);
	return *p;
}

int test13(int *q) {
	int *p = q;
	idc(p);
	return *p;
}

int test21(int *q, int *x) {
	if (q)
		;
	if (x)
		;
	int *p = q;
	return *p; // expected-warning{{Dereference of null pointer}}
}

int test22(int *q, int *x) {
  idc(q);
	if (x)
		;
	int *p = q;
	return *p;
}

int test23(int *q, int *x) {
  idc(q);
	if (x)
		;
	int *p = q;
  if (!p)
    ;
	return *p; // False negative
}

void use(char *p) {
  if (!p)
    return;
  p[0] = 'a';
}

void test24(char *buffer) {
  use(buffer);
  buffer[1] = 'b';
}

// Ensure idc works on pointers with constant offset.
void idcchar(const char *s2) {
  if(s2)
    ;
}
void testConstantOffset(char *value) {
  char *cursor = value + 5;
  idcchar(cursor);
  if (*cursor) {
    cursor++;
  }
}

// Ensure idc works for integer zero values (ex: suppressed div by zero).
void idcZero(int assume) {
  if (assume)
    ;
}

int idcTriggerZeroValue(int m) {
  idcZero(m);
  return 5/m; // no-warning
}

int idcTriggerZeroValueThroughCall(int i) {
  return 5/i; // no-warning
}
void idcTrackZeroValueThroughCall(int x) {
  idcZero(x);
  idcTriggerZeroValueThroughCall(x);
}

int idcTriggerZeroThroughDoubleAssignemnt(int i) {
  return 5/i; // no-warning
}
void idcTrackZeroThroughDoubleAssignemnt(int x) {
  idcZero(x);
  int y = x;
  int z = y;
  idcTriggerZeroValueThroughCall(z);
}

struct S {
  int f1;
  int f2;
};

void idcTrackZeroValueThroughUnaryPointerOperators(struct S *s) {
  idc(s);
  *(&(s->f1)) = 7; // no-warning
}

void idcTrackZeroValueThroughUnaryPointerOperatorsWithOffset1(struct S *s) {
  idc(s);
  int *x = &(s->f2);
  *x = 7; // no-warning
}

void idcTrackZeroValueThroughUnaryPointerOperatorsWithOffset2(struct S *s) {
  idc(s);
  int *x = &(s->f2) - 1;
  // FIXME: Should not warn.
  *x = 7; // expected-warning{{Dereference of null pointer}}
}

void idcTrackZeroValueThroughUnaryPointerOperatorsWithAssignment(struct S *s) {
  idc(s);
  int *x = &(s->f1);
  *x = 7; // no-warning
}

void idcTrackZeroValueThroughManyUnaryPointerOperatorsWithAssignment(struct S *s) {
  idc(s);
  int *x = &*&(s->f1);
  *x = 7; // no-warning
}

void idcTrackZeroValueThroughManyUnaryPointerOperatorsWithAssignmentAndUnaryIncrement(struct S *s) {
  idc(s);
  int *x = &*&((++s)->f1);
  *x = 7; // no-warning
}


struct S2 {
  int a[1];
};

void idcTrackZeroValueThroughUnaryPointerOperatorsWithArrayField(struct S2 *s) {
  idc(s);
  *(&(s->a[0])) = 7; // no-warning
}

void idcTrackConstraintThroughSymbolicRegion(int **x) {
  idc(*x);
  // FIXME: Should not warn.
  **x = 7; // expected-warning{{Dereference of null pointer}}
}

void idcTrackConstraintThroughSymbolicRegionAndParens(int **x) {
  idc(*x);
  // FIXME: Should not warn.
  *(*x) = 7; // expected-warning{{Dereference of null pointer}}
}

int *idcPlainNull(int coin) {
  if (coin)
    return 0;
  static int X;
  return &X;
}

void idcTrackZeroValueThroughSymbolicRegion(int coin, int **x) {
  *x = idcPlainNull(coin);
  **x = 7; // no-warning
}

void idcTrackZeroValueThroughSymbolicRegionAndParens(int coin, int **x) {
  *x = idcPlainNull(coin);
  *(*x) = 7; // no-warning
}

struct WithInt {
  int i;
};

struct WithArray {
  struct WithInt arr[1];
};

struct WithArray *idcPlainNullWithArray(int coin) {
  if (coin)
    return 0;
  static struct WithArray S;
  return &S;
}

void idcTrackZeroValueThroughSymbolicRegionWithArray(int coin, struct WithArray **s) {
  *s = idcPlainNullWithArray(coin);
  (*s)->arr[0].i = 1; // no-warning
  // Same thing.
  (*s)->arr->i = 1; // no-warning
}

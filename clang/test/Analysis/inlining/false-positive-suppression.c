// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-config suppress-null-return-paths=false -verify %s
// RUN: %clang_cc1 -analyze -analyzer-checker=core -verify -DSUPPRESSED=1 %s
// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-config avoid-suppressing-null-argument-paths=true -DSUPPRESSED=1 -DNULL_ARGS=1 -verify %s

int opaquePropertyCheck(void *object);
int coin();

int *getNull() {
  return 0;
}

int *dynCastToInt(void *ptr) {
  if (opaquePropertyCheck(ptr))
    return (int *)ptr;
  return 0;
}

int *dynCastOrNull(void *ptr) {
  if (!ptr)
    return 0;
  if (opaquePropertyCheck(ptr))
    return (int *)ptr;
  return 0;
}


void testDynCast(void *p) {
  int *casted = dynCastToInt(p);
  *casted = 1;
#ifndef SUPPRESSED
  // expected-warning@-2 {{Dereference of null pointer}}
#endif
}

void testDynCastOrNull(void *p) {
  int *casted = dynCastOrNull(p);
  *casted = 1;
#ifndef SUPPRESSED
  // expected-warning@-2 {{Dereference of null pointer}}
#endif
}


void testBranch(void *p) {
  int *casted;

  // Although the report will be suppressed on one branch, it should still be
  // valid on the other.
  if (coin()) {
    casted = dynCastToInt(p);
  } else {
    if (p)
      return;
    casted = (int *)p;
  }

  *casted = 1; // expected-warning {{Dereference of null pointer}}
}

void testBranchReversed(void *p) {
  int *casted;

  // Although the report will be suppressed on one branch, it should still be
  // valid on the other.
  if (coin()) {
    if (p)
      return;
    casted = (int *)p;
  } else {
    casted = dynCastToInt(p);
  }

  *casted = 1; // expected-warning {{Dereference of null pointer}}
}


// --------------------------
// "Suppression suppression"
// --------------------------

void testDynCastOrNullOfNull() {
  // Don't suppress when one of the arguments is NULL.
  int *casted = dynCastOrNull(0);
  *casted = 1;
#if !SUPPRESSED || NULL_ARGS
  // expected-warning@-2 {{Dereference of null pointer}}
#endif
}

void testDynCastOfNull() {
  // Don't suppress when one of the arguments is NULL.
  int *casted = dynCastToInt(0);
  *casted = 1;
#if !SUPPRESSED || NULL_ARGS
  // expected-warning@-2 {{Dereference of null pointer}}
#endif
}

int *lookUpInt(int unused) {
  if (coin())
    return 0;
  static int x;
  return &x;
}

void testZeroIsNotNull() {
  // /Do/ suppress when the argument is 0 (an integer).
  int *casted = lookUpInt(0);
  *casted = 1;
#ifndef SUPPRESSED
  // expected-warning@-2 {{Dereference of null pointer}}
#endif
}

void testTrackNull() {
  // /Do/ suppress if the null argument came from another call returning null.
  int *casted = dynCastOrNull(getNull());
  *casted = 1;
#ifndef SUPPRESSED
  // expected-warning@-2 {{Dereference of null pointer}}
#endif
}

void testTrackNullVariable() {
  // /Do/ suppress if the null argument came from another call returning null.
  int *ptr;
  ptr = getNull();
  int *casted = dynCastOrNull(ptr);
  *casted = 1;
#ifndef SUPPRESSED
  // expected-warning@-2 {{Dereference of null pointer}}
#endif
}


// ---------------------------------------
// FALSE NEGATIVES (over-suppression)
// ---------------------------------------

void testNoArguments() {
  // In this case the function has no branches, and MUST return null.
  int *casted = getNull();
  *casted = 1;
#ifndef SUPPRESSED
  // expected-warning@-2 {{Dereference of null pointer}}
#endif
}

int *getNullIfNonNull(void *input) {
  if (input)
    return 0;
  static int x;
  return &x;
}

void testKnownPath(void *input) {
  if (!input)
    return;

  // In this case we have a known value for the argument, and thus the path
  // through the function doesn't ever split.
  int *casted = getNullIfNonNull(input);
  *casted = 1;
#ifndef SUPPRESSED
  // expected-warning@-2 {{Dereference of null pointer}}
#endif
}

int *alwaysReturnNull(void *input) {
  if (opaquePropertyCheck(input))
    return 0;
  return 0;
}

void testAlwaysReturnNull(void *input) {
  // In this case all paths out of the function return 0, but they are all
  // dominated by a branch whose condition we don't know!
  int *casted = alwaysReturnNull(input);
  *casted = 1;
#ifndef SUPPRESSED
  // expected-warning@-2 {{Dereference of null pointer}}
#endif
}


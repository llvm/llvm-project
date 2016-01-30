// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-config suppress-null-return-paths=false -verify %s
// RUN: %clang_cc1 -analyze -analyzer-checker=core -verify -DSUPPRESSED=1 %s
// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-config avoid-suppressing-null-argument-paths=true -DSUPPRESSED=1 -DNULL_ARGS=1 -verify %s

int opaquePropertyCheck(void *object);
int coin();

int *getNull() {
  return 0;
}

int* getPtr();

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

void testMultipleStore(void *p) {
  int *casted = 0;
  casted = dynCastToInt(p);
  *casted = 1;
#ifndef SUPPRESSED
  // expected-warning@-2 {{Dereference of null pointer}}
#endif
}

// Test that div by zero does not get suppressed. This is a policy choice.
int retZero() {
  return 0;
}
int triggerDivZero () {
  int y = retZero();
  return 5/y; // expected-warning {{Division by zero}}
}

// Treat a function-like macro similarly to an inlined function, so suppress
// warnings along paths resulting from inlined checks.
#define MACRO_WITH_CHECK(a) ( ((a) != 0) ? *a : 17)
void testInlineCheckInMacro(int *p) {
  int i = MACRO_WITH_CHECK(p);
  (void)i;

  *p = 1; // no-warning
}

#define MACRO_WITH_NESTED_CHECK(a) ( { int j = MACRO_WITH_CHECK(a); j; } )
void testInlineCheckInNestedMacro(int *p) {
  int i = MACRO_WITH_NESTED_CHECK(p);
  (void)i;

  *p = 1; // no-warning
}

// If there is a check in a macro that is not function-like, don't treat
// it like a function so don't suppress.
#define NON_FUNCTION_MACRO_WITH_CHECK ( ((p) != 0) ? *p : 17)
void testNonFunctionMacro(int *p) {
  int i = NON_FUNCTION_MACRO_WITH_CHECK ;
  (void)i;

  *p = 1; // expected-warning {{Dereference of null pointer (loaded from variable 'p')}}
}


// This macro will dereference its argument if the argument is NULL.
#define MACRO_WITH_ERROR(a) ( ((a) != 0) ? 0 : *a)
void testErrorInMacro(int *p) {
  int i = MACRO_WITH_ERROR(p); // expected-warning {{Dereference of null pointer (loaded from variable 'p')}}
  (void)i;
}

// Here the check (the "if") is not in a macro, so we should still warn.
#define MACRO_IN_GUARD(a) (!(a))
void testMacroUsedAsGuard(int *p) {
  if (MACRO_IN_GUARD(p))
    *p = 1; // expected-warning {{Dereference of null pointer (loaded from variable 'p')}}
}

// When a nil case split is introduced in a macro and the macro is in a guard,
// we still shouldn't warn.
int isNull(int *p);
int isEqual(int *p, int *q);
#define ISNULL(ptr)    ((ptr) == 0 || isNull(ptr))
#define ISEQUAL(a, b)    ((int *)(a) == (int *)(b) || (ISNULL(a) && ISNULL(b)) || isEqual(a,b))
#define ISNOTEQUAL(a, b)   (!ISEQUAL(a, b))
void testNestedDisjunctiveMacro(int *p, int *q) {
  if (ISNOTEQUAL(p,q)) {
    (void)*p; // no-warning
    (void)*q; // no-warning
  }

  (void)*p; // no-warning
  (void)*q; // no-warning
}

// Here the check is entirely in non-macro code even though the code itself
// is a macro argument.
#define MACRO_DO_IT(a) (a)
void testErrorInArgument(int *p) {
  int i = MACRO_DO_IT((p ? 0 : *p)); // expected-warning {{Dereference of null pointer (loaded from variable 'p')}}c
  (void)i;
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

void inlinedIsDifferent(int inlined) {
  int i;

  // We were erroneously picking up the inner stack frame's initialization,
  // even though the error occurs in the outer stack frame!
  int *p = inlined ? &i : getNull();

  if (!inlined)
    inlinedIsDifferent(1);

  *p = 1;
#ifndef SUPPRESSED
  // expected-warning@-2 {{Dereference of null pointer}}
#endif
}

void testInlinedIsDifferent() {
  // <rdar://problem/13787723>
  inlinedIsDifferent(0);
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

int derefArg(int *p) {
	return *p;
#ifndef SUPPRESSED
  // expected-warning@-2 {{Dereference of null pointer}}
#endif
}
void ternaryArg(char cond) {
	static int x;
	derefArg(cond ? &x : getNull());
}

int derefArgCast(char *p) {
	return *p;
#ifndef SUPPRESSED
  // expected-warning@-2 {{Dereference of null pointer}}
#endif
}
void ternaryArgCast(char cond) {
	static int x;
	derefArgCast((char*)((unsigned)cond ? &x : getNull()));
}

int derefAssignment(int *p) {
	return *p;
#ifndef SUPPRESSED
  // expected-warning@-2 {{Dereference of null pointer}}
#endif
}

void ternaryAssignment(char cond) {
  static int x;
  int *p = cond ? getNull() : getPtr();
  derefAssignment(p);
}

int *retNull(char cond) {
  static int x;
  return cond ? &x : getNull();
}
int ternaryRetNull(char cond) {
  int *p = retNull(cond);
  return *p;
#ifndef SUPPRESSED
  // expected-warning@-2 {{Dereference of null pointer}}
#endif
}

// Test suppression of nested conditional operators.
int testConditionalOperatorSuppress(int x) {
  return *(x ? getNull() : getPtr());
#ifndef SUPPRESSED
  // expected-warning@-2 {{Dereference of null pointer}}
#endif
}
int testNestedConditionalOperatorSuppress(int x) {
  return *(x ? (x ? getNull() : getPtr()) : getPtr());
#ifndef SUPPRESSED
  // expected-warning@-2 {{Dereference of null pointer}}
#endif
}
int testConditionalOperator(int x) {
  return *(x ? 0 : getPtr()); // expected-warning {{Dereference of null pointer}}
}
int testNestedConditionalOperator(int x) {
  return *(x ? (x ? 0 : getPtr()) : getPtr()); // expected-warning {{Dereference of null pointer}}
}

int testConditionalOperatorSuppressFloatCond(float x) {
  return *(x ? getNull() : getPtr());
#ifndef SUPPRESSED
  // expected-warning@-2 {{Dereference of null pointer}}
#endif
}


// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-output=text -analyzer-config suppress-null-return-paths=false -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-output=plist-multi-file -analyzer-config suppress-null-return-paths=false %s -o %t.plist
// RUN: %normalize_plist <%t.plist | diff -ub %S/Inputs/expected-plists/path-notes.c.plist -

void zero(int **p) {
  *p = 0;
  // expected-note@-1 {{Null pointer value stored to 'a'}}
}

void testZero(int *a) {
  zero(&a);
  // expected-note@-1 {{Calling 'zero'}}
  // expected-note@-2 {{Returning from 'zero'}}
  *a = 1; // expected-warning{{Dereference of null pointer}}
  // expected-note@-1 {{Dereference of null pointer (loaded from variable 'a')}}
}

void testCheck(int *a) {
  if (a) {
    // expected-note@-1 + {{Assuming 'a' is null}}
    // expected-note@-2 + {{Taking false branch}}
    ;
  }
  *a = 1; // expected-warning{{Dereference of null pointer}}
  // expected-note@-1 {{Dereference of null pointer (loaded from variable 'a')}}
}


int *getPointer(void);

void testInitCheck(void) {
  int *a = getPointer();
  // expected-note@-1 {{'a' initialized here}}
  if (a) {
    // expected-note@-1 + {{Assuming 'a' is null}}
    // expected-note@-2 + {{Taking false branch}}
    ;
  }
  *a = 1; // expected-warning{{Dereference of null pointer}}
  // expected-note@-1 {{Dereference of null pointer (loaded from variable 'a')}}
}

void testStoreCheck(int *a) {
  a = getPointer();
  // expected-note@-1 {{Value assigned to 'a'}}
  if (a) {
    // expected-note@-1 + {{Assuming 'a' is null}}
    // expected-note@-2 + {{Taking false branch}}
    ;
  }
  *a = 1; // expected-warning{{Dereference of null pointer}}
  // expected-note@-1 {{Dereference of null pointer (loaded from variable 'a')}}
}


int *getZero(void) {
  int *p = 0;
  // expected-note@-1 + {{'p' initialized to a null pointer value}}
  // ^ This note checks that we add a second visitor for the return value.
  return p;
  // expected-note@-1 + {{Returning null pointer (loaded from 'p')}}
}

void testReturnZero(void) {
  *getZero() = 1; // expected-warning{{Dereference of null pointer}}
  // expected-note@-1 {{Calling 'getZero'}}
  // expected-note@-2 {{Returning from 'getZero'}}
  // expected-note@-3 {{Dereference of null pointer}}
}

int testReturnZero2(void) {
  return *getZero(); // expected-warning{{Dereference of null pointer}}
  // expected-note@-1 {{Calling 'getZero'}}
  // expected-note@-2 {{Returning from 'getZero'}}
  // expected-note@-3 {{Dereference of null pointer}}
}

void testInitZero(void) {
  int *a = getZero();
  // expected-note@-1 {{Calling 'getZero'}}
  // expected-note@-2 {{Returning from 'getZero'}}
  // expected-note@-3 {{'a' initialized to a null pointer value}}
  *a = 1; // expected-warning{{Dereference of null pointer}}
  // expected-note@-1 {{Dereference of null pointer (loaded from variable 'a')}}
}

void testStoreZero(int *a) {
  a = getZero();
  // expected-note@-1 {{Calling 'getZero'}}
  // expected-note@-2 {{Returning from 'getZero'}}
  // expected-note@-3 {{Null pointer value stored to 'a'}}
  *a = 1; // expected-warning{{Dereference of null pointer}}
  // expected-note@-1 {{Dereference of null pointer (loaded from variable 'a')}}
}

void usePointer(int *p) {
  *p = 1; // expected-warning{{Dereference of null pointer}}
  // expected-note@-1 {{Dereference of null pointer}}
}

void testUseOfNullPointer(void) {
  // Test the case where an argument expression is itself a call.
  usePointer(getZero());
  // expected-note@-1 {{Calling 'getZero'}}
  // expected-note@-2 {{Returning from 'getZero'}}
  // expected-note@-3 {{Passing null pointer value via 1st parameter 'p'}}
  // expected-note@-4 {{Calling 'usePointer'}}
}

struct X { char *p; };

void setFieldToNull(struct X *x) {
	x->p = 0; // expected-note {{Null pointer value stored to field 'p'}}
}

int testSetFieldToNull(struct X *x) {
  setFieldToNull(x); // expected-note {{Calling 'setFieldToNull'}}
                     // expected-note@-1{{Returning from 'setFieldToNull'}}
  return *x->p;
  // expected-warning@-1 {{Dereference of null pointer (loaded from field 'p')}}
  // expected-note@-2 {{Dereference of null pointer (loaded from field 'p')}}
}

struct Outer {
  struct Inner {
    int *p;
  } inner;
};

void test(struct Outer *wrapperPtr) {
  wrapperPtr->inner.p = 0;  // expected-note {{Null pointer value stored to field 'p'}}
  *wrapperPtr->inner.p = 1; //expected-warning {{Dereference of null pointer (loaded from field 'p')}}
                            // expected-note@-1 {{Dereference of null pointer (loaded from field 'p')}}
}

void test4(int **p) {
  if (*p) return; // expected-note {{Taking false branch}}
                  // expected-note@-1 {{Assuming pointer value is null}}
  **p = 1; // expected-warning {{Dereference of null pointer}}
           // expected-note@-1 {{Dereference of null pointer}}
}

void boringCallee(void) {
}

void interestingCallee(int *x) {
  *x = 0; // expected-note{{The value 0 is assigned to 'x'}}
  boringCallee(); // no-note
}

int testBoringCalleeOfInterestingCallee(void) {
  int x;
  interestingCallee(&x); // expected-note{{Calling 'interestingCallee'}}
                         // expected-note@-1{{Returning from 'interestingCallee'}}
  return 1 / x; // expected-warning{{Division by zero}}
                // expected-note@-1{{Division by zero}}
}


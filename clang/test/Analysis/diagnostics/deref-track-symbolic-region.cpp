// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-output=text -verify %s

struct S {
  int *x;
  int y;
};

S &getSomeReference();
void test(S *p) {
  S &r = *p;   //expected-note {{'r' initialized here}}
  if (p) return;
               //expected-note@-1{{Taking false branch}}
               //expected-note@-2{{Assuming 'p' is null}}
  r.y = 5; // expected-warning {{Access to field 'y' results in a dereference of a null pointer (loaded from variable 'r')}}
           // expected-note@-1{{Access to field 'y' results in a dereference of a null pointer (loaded from variable 'r')}}
}

void testRefParam(int *ptr) {
	int &ref = *ptr; // expected-note {{'ref' initialized here}}
	if (ptr)
    // expected-note@-1{{Assuming 'ptr' is null}}
    // expected-note@-2{{Taking false branch}}
		return;

	extern void use(int &ref);
	use(ref); // expected-warning{{Forming reference to null pointer}}
            // expected-note@-1{{Forming reference to null pointer}}
}

int testRefToNullPtr() {
  int *p = 0;         // expected-note {{'p' initialized to a null pointer value}}
  int *const &p2 = p; // expected-note{{'p2' initialized here}}
  int *p3 = p2;       // expected-note {{'p3' initialized to a null pointer value}}
  return *p3;         // expected-warning {{Dereference of null pointer}}
                      // expected-note@-1{{Dereference of null pointer}}
}

int testRefToNullPtr2() {
  int *p = 0;         // expected-note {{'p' initialized to a null pointer value}}
  int *const &p2 = p; // expected-note{{'p2' initialized here}}
  return *p2;         //expected-warning {{Dereference of null pointer}}
                      // expected-note@-1{{Dereference of null pointer}}
}

void testMemberNullPointerDeref() {
  struct Wrapper {char c; int *ptr; };  
  Wrapper w = {'a', nullptr};           // expected-note {{'w.ptr' initialized to a null pointer value}}
  *w.ptr = 1;                           //expected-warning {{Dereference of null pointer}}
                                        // expected-note@-1{{Dereference of null pointer}}
}

void testMemberNullReferenceDeref() {
  struct Wrapper {char c; int &ref; };
  Wrapper w = {.c = 'a', .ref = *(int *)0 }; // expected-note {{'w.ref' initialized to a null pointer value}}
                                             // expected-warning@-1 {{binding dereferenced null pointer to reference has undefined behavior}}
  w.ref = 1;                                 //expected-warning {{Dereference of null pointer}}
                                             // expected-note@-1{{Dereference of null pointer}}
}

void testReferenceToPointerWithNullptr() {
  int *i = nullptr;                   // expected-note {{'i' initialized to a null pointer value}}
  struct Wrapper {char c; int *&a;};
  Wrapper w {'c', i};                 // expected-note{{'w.a' initialized here}}
  *(w.a) = 25;                        // expected-warning {{Dereference of null pointer}}
                                      // expected-note@-1 {{Dereference of null pointer}}
}

void testNullReferenceToPointer() {
  struct Wrapper {char c; int *&a;};
  Wrapper w {'c', *(int **)0 };           // expected-note{{'w.a' initialized to a null pointer value}}
                                          // expected-warning@-1 {{binding dereferenced null pointer to reference has undefined behavior}}
  w.a = nullptr;                          // expected-warning {{Dereference of null pointer}}
                                          // expected-note@-1 {{Dereference of null pointer}}
}
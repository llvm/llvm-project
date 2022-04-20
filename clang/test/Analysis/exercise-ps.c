// RUN: %clang_analyze_cc1 %s -verify -Wno-error=implicit-function-declaration \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config core.CallAndMessage:ArgPointeeInitializedness=true
//
// Just exercise the analyzer on code that has at one point caused issues
// (i.e., no assertions or crashes).

static void f1(const char *x, char *y) {
  while (*x != 0) {
    *y++ = *x++;
  }
}

// This following case checks that we properly handle typedefs when getting
// the RvalueType of an ElementRegion.
typedef struct F12_struct {} F12_typedef;
typedef void* void_typedef;
void_typedef f2_helper(void);
static void f2(void *buf) {
  F12_typedef* x;
  x = f2_helper();
  memcpy((&x[1]), (buf), 1); // expected-warning{{call to undeclared library function 'memcpy' with type 'void *(void *, const void *}} \
  // expected-note{{include the header <string.h> or explicitly provide a declaration for 'memcpy'}}
}

// AllocaRegion is untyped. Void pointer isn't of much help either. Before
// realizing that the value is undefined, we need to somehow figure out
// what type of value do we expect.
void f3(void *dest) {
  void *src = __builtin_alloca(5);
  memcpy(dest, src, 1); // expected-warning{{2nd function call argument is a pointer to uninitialized value}}
}

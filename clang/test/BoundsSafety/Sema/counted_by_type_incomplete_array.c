
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
#include <ptrcheck.h>
// Test incomplete arrays

//==============================================================================
// Extern declaration that is later redefined
//==============================================================================


extern int external_arr_len;
// expected-error@+1{{cannot apply '__counted_by' attribute to 'char (*)[]' because 'char[]' has unknown size; did you mean to use '__sized_by' instead?}}
extern char (* __counted_by(external_arr_len) incompleteArrayPtr)[]; // expected-note{{'incompleteArrayPtr' declared here}}
void use_incompleteArrayPtr_when_incomplete(void) {
    // expected-error@+1{{subscript of pointer to incomplete type 'char[]'}}
    char x = incompleteArrayPtr[0][0];
}

// Provide a complete definition with the array size defined
int external_arr_len;
// expected-error@+1{{conflicting '__counted_by' attribute with the previous variable declaration}}
char (* __counted_by(external_arr_len) incompleteArrayPtr)[4];

void use_incompleteArrayPtr_when_complete(void) {
    char x = incompleteArrayPtr[0][0]; // OK?
}

int external_arr_len2;
char (* __counted_by(external_arr_len2) CompleteArrayPtr)[4]; // OK

//==============================================================================
// Global vars
//==============================================================================

int global_arr_len;
// expected-error@+1{{cannot apply '__counted_by' attribute to 'char (*)[]' because 'char[]' has unknown size; did you mean to use '__sized_by' instead?}}
char (* __counted_by(global_arr_len) GlobalCBIncompleteArrayPtr)[];
// expected-error@+1{{cannot apply '__counted_by_or_null' attribute to 'char (*)[]' because 'char[]' has unknown size; did you mean to use '__sized_by_or_null' instead?}}
char (* __counted_by_or_null(global_arr_len) GlobalCBONIncompleteArrayPtr)[];

char (* __counted_by(global_arr_len) GlobalCBCompleteArrayPtr)[2]; // OK
char (* __counted_by_or_null(global_arr_len) GlobalCBONCompleteArrayPtr)[2]; // OK

char (* __counted_by(0) GlobalCBCompleteArrayPtrZeroCount)[2]; // OK
// expected-error@+1{{implicitly initializing 'GlobalCBCompleteArrayPtrOneCount' of type 'char (*__single __counted_by(1))[2]' (aka 'char (*__single)[2]') and count value of 1 with null always fails}}
char (* __counted_by(1) GlobalCBCompleteArrayPtrOneCount)[2];

//==============================================================================
// Local vars
//==============================================================================

void local_cb(void) {
  int size = 0;
  // expected-error@+1{{cannot apply '__counted_by' attribute to 'char (*)[]' because 'char[]' has unknown size; did you mean to use '__sized_by' instead?}}
  char (* __counted_by(size) local_implicit_init)[];

  // expected-error@+1{{subscript of pointer to incomplete type 'char[]'}}
  local_implicit_init[0];

  int size2 = 0;
  // expected-error@+1{{cannot apply '__counted_by' attribute to 'char (*)[]' because 'char[]' has unknown size; did you mean to use '__sized_by' instead?}}
  char (* __counted_by(size2) local_explicit_init)[] = 0x0;
}

void local_cbon(void) {
  int size = 0;
  // expected-error@+1{{cannot apply '__counted_by_or_null' attribute to 'char (*)[]' because 'char[]' has unknown size; did you mean to use '__sized_by_or_null' instead?}}
  char (* __counted_by_or_null(size) local_implicit_init)[];

  // expected-error@+1{{subscript of pointer to incomplete type 'char[]'}}
  local_implicit_init[0];

  int size2 = 0;
  // expected-error@+1{{cannot apply '__counted_by_or_null' attribute to 'char (*)[]' because 'char[]' has unknown size; did you mean to use '__sized_by_or_null' instead?}}
  char (* __counted_by_or_null(size2) local_explicit_init)[] = 0x0;
}

//==============================================================================
// Parameters
//==============================================================================

// expected-error@+1{{cannot apply '__counted_by' attribute to 'char (*)[]' because 'char[]' has unknown size; did you mean to use '__sized_by' instead?}}
void decl_param_cb(char (* __counted_by(size) p)[], int size) {}
// expected-error@+1{{cannot apply '__counted_by_or_null' attribute to 'char (*)[]' because 'char[]' has unknown size; did you mean to use '__sized_by_or_null' instead?}}
void decl_param_cbon(char (* __counted_by_or_null(size) p)[], int size) {}

// expected-error@+1{{cannot apply '__counted_by' attribute to 'char (*)[]' because 'char[]' has unknown size; did you mean to use '__sized_by' instead?}}
void access_param_cb(char (* __counted_by(size) p)[], int size) {
  void* local = p;

  // expected-error@+1{{subscript of pointer to incomplete type 'char[]'}}
  p[0];
  // expected-error@+1{{subscript of pointer to incomplete type 'char[]'}}
  p[0][0];

  // expected-error@+1{{subscript of pointer to incomplete type 'char[]'}}
  &p[0];

  p = 0; // OK
  size = 0; // OK
}

// expected-error@+1{{cannot apply '__counted_by_or_null' attribute to 'char (*)[]' because 'char[]' has unknown size; did you mean to use '__sized_by_or_null' instead?}}
void access_param_cbon(char (* __counted_by_or_null(size) p)[], int size) {
  // This doesn't show up as an error because error recovery for treats `p` as
  // having type `char (*__single __sized_by_or_null(size))[]`
  void* local = p;

  // expected-error@+1{{subscript of pointer to incomplete type 'char[]'}}
  p[0];
  // expected-error@+1{{subscript of pointer to incomplete type 'char[]'}}
  p[0][0];

  // expected-error@+1{{subscript of pointer to incomplete type 'char[]'}}
  &p[0];

  p = 0; // OK
  size = 0;
}

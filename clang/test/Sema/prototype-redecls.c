// RUN: %clang_cc1 -fsyntax-only -Wno-strict-prototypes -verify %s

void blapp(int); // expected-note {{previous}}
void blapp() { } // expected-error {{conflicting types for 'blapp'}}

void yarp(int, ...); // expected-note {{previous}}
void yarp();         // expected-error {{conflicting types for 'yarp'}}

void blarg(int, ...); // expected-note {{previous}}
void blarg() {}       // expected-error {{conflicting types for 'blarg'}}

void blerp(short);      // expected-note {{previous}}
void blerp(x) int x; {} // expected-error {{conflicting types for 'blerp'}}

void glerp(int);
void glerp(x) short x; {} // Okay, promoted type is fine

// All these cases are okay
void derp(int);
void derp(x) int x; {}

void garp(int);
void garp();
void garp(x) int x; {}

// Ensure redeclarations that conflict with a builtin use a note which makes it
// clear that the previous declaration was a builtin.
float rintf() { // expected-error {{conflicting types for 'rintf'}} \
                   expected-note {{'rintf' is a builtin with type 'float (float)'}}
  return 1.0f;
}

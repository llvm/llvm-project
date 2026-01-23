// RUN: %clang_cc1 -verify=expected,c-expected -std=c2y -Wall -pedantic %s
// RUN: %clang_cc1 -verify=expected,c-expected,ped -std=c23 -Wall -pedantic %s
// RUN: %clang_cc1 -verify=expected,cxx-expected,gnu -Wall -pedantic -x c++ %s
// RUN: %clang_cc1 -verify=expected,c-expected,pre -std=c2y -Wpre-c2y-compat -Wall -pedantic %s

/* WG14 N3370: Yes
 * Case range expressions v3.1
 *
 * This introduces the ability to specify closed ranges in case statements in a
 * switch statement. This was already a well-supported Clang extension before
 * it was standardized.
 */

void correct(int i) {
  constexpr int j = 100, k = 200;
  switch (i) {
  case 12 ... 14: break; /* gnu-warning {{case ranges are a GNU extension}}
                            ped-warning {{case ranges are a C2y extension}}
                            pre-warning {{case ranges are incompatible with C standards before C2y}}
                          */
  // Implementations are encouraged to diagnose empty ranges.
  case 15 ... 11: break;  /* expected-warning {{empty case range specified}}
                             gnu-warning {{case ranges are a GNU extension}}
                             ped-warning {{case ranges are a C2y extension}}
                             pre-warning {{case ranges are incompatible with C standards before C2y}}
                           */
  // This is not an empty range, it's a range of a single value.
  case 10 ... 10: break; /* gnu-warning {{case ranges are a GNU extension}}
                            ped-warning {{case ranges are a C2y extension}}
                            pre-warning {{case ranges are incompatible with C standards before C2y}}
                          */
  case j ... k: break;   /* gnu-warning {{case ranges are a GNU extension}}
                            ped-warning {{case ranges are a C2y extension}}
                            pre-warning {{case ranges are incompatible with C standards before C2y}}
                          */
  }
}

void incorrect(int i) { // cxx-expected-note 2 {{declared here}}
  switch (i) {
  // The values have to be integer constant expressions. Note that when the
  // initial value in the range is an error, we don't issue the warnings about
  // extensions or incompatibility.
  case i ... 10: break;    /* c-expected-error {{expression is not an integer constant expression}}
                              cxx-expected-error {{case value is not a constant expression}}
                              cxx-expected-note {{function parameter 'i' with unknown value cannot be used in a constant expression}}
                            */
  case 10 ... i: break;    /* c-expected-error {{expression is not an integer constant expression}}
                              cxx-expected-error {{case value is not a constant expression}}
                              cxx-expected-note {{function parameter 'i' with unknown value cannot be used in a constant expression}}
                              gnu-warning {{case ranges are a GNU extension}}
                              ped-warning {{case ranges are a C2y extension}}
                              pre-warning {{case ranges are incompatible with C standards before C2y}}
                            */
  case 1.3f ... 10: break; /* c-expected-error {{integer constant expression must have integer type, not 'float'}}
                              cxx-expected-error {{conversion from 'float' to 'int' is not allowed in a converted constant expression}}
                            */
  case 10 ... "a": break;  /* c-expected-error {{integer constant expression must have integer type, not 'char[2]'}}
                              cxx-expected-error {{value of type 'const char[2]' is not implicitly convertible to 'int'}}
                              gnu-warning {{case ranges are a GNU extension}}
                              ped-warning {{case ranges are a C2y extension}}
                              pre-warning {{case ranges are incompatible with C standards before C2y}}
                            */
  }

  switch (i) {
  // Cannot have multiple cases covering the same value.
  // FIXME: diagnostic quality here is poor. The "previous case" note is
  // showing up on a subsequent line (I'd expect the error and note to be
  // reversed), and "duplicate case value 20" is showing up on a line where
  // there is no duplicate value 20 to begin with.
  case 10 ... 20: break; /* expected-error {{duplicate case value '11'}}
                            expected-note {{previous case defined here}}
                            gnu-warning {{case ranges are a GNU extension}}
                            ped-warning {{case ranges are a C2y extension}}
                            pre-warning {{case ranges are incompatible with C standards before C2y}}
                          */
  case 11: break;        /* expected-note {{previous case defined here}}
                          */
  case 11 ... 14: break; /* expected-error {{duplicate case value '20'}}
                            gnu-warning {{case ranges are a GNU extension}}
                            ped-warning {{case ranges are a C2y extension}}
                            pre-warning {{case ranges are incompatible with C standards before C2y}}
                          */
  }

  // The values specified by the range shall not change as a result of
  // conversion to the promoted type of the controlling expression.
  // FIXME: the overflow warnings seem like they probably should also trigger
  // in C++ as they do in C.
  switch ((unsigned char)i) {
  case 254 ... 256: break; /* c-expected-warning {{overflow converting case value to switch condition type (256 to 0)}}
                              gnu-warning {{case ranges are a GNU extension}}
                              ped-warning {{case ranges are a C2y extension}}
                              pre-warning {{case ranges are incompatible with C standards before C2y}}
                            */
  case 257 ... 258: break; /* c-expected-warning {{overflow converting case value to switch condition type (257 to 1)}}
                              c-expected-warning {{overflow converting case value to switch condition type (258 to 2)}}
                              gnu-warning {{case ranges are a GNU extension}}
                              ped-warning {{case ranges are a C2y extension}}
                              pre-warning {{case ranges are incompatible with C standards before C2y}}
                            */
  }
}


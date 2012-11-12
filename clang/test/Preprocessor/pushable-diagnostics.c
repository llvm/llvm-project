// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s

#pragma clang diagnostic pop // expected-warning{{pragma diagnostic pop could not pop, no matching push}}

#pragma clang diagnostic puhs // expected-warning {{pragma diagnostic expected 'error', 'warning', 'ignored', 'fatal', 'push', or 'pop'}}

int a = 'df'; // expected-warning{{multi-character character constant}}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmultichar"

int b = 'df'; // no warning.
#pragma clang diagnostic pop

int c = 'df';  // expected-warning{{multi-character character constant}}

#pragma clang diagnostic pop // expected-warning{{pragma diagnostic pop could not pop, no matching push}}

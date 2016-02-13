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

// Test -Weverything

void ppo0(){} // first verify that we do not give anything on this
#pragma clang diagnostic push // now push

#pragma clang diagnostic warning "-Weverything" 
void ppr1(){} // expected-warning {{no previous prototype for function 'ppr1'}}

#pragma clang diagnostic push // push again
#pragma clang diagnostic ignored "-Weverything"  // Set to ignore in this level.
void pps2(){}
#pragma clang diagnostic warning "-Weverything"  // Set to warning in this level.
void ppt2(){} // expected-warning {{no previous prototype for function 'ppt2'}}
#pragma clang diagnostic error "-Weverything"  // Set to error in this level.
void ppt3(){} // expected-error {{no previous prototype for function 'ppt3'}}
#pragma clang diagnostic pop // pop should go back to warning level

void pps1(){} // expected-warning {{no previous prototype for function 'pps1'}}


#pragma clang diagnostic pop // Another pop should disble it again
void ppu(){}


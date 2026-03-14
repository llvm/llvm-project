// RUN: %clang_cc1 -verify %s

// This case reproduces a problem that is shown here:
// https://bugs.llvm.org/show_bug.cgi?id=46540
// No assertion should happen during printing of diagnostic messages.

// expected-error@13{{'b' does not refer to a type name in pseudo-destructor expression; expected the name of type 'volatile long'}}
// expected-error@13{{expected ')'}}
// expected-note@13{{to match this '('}}
// expected-error@13{{reference to pseudo-destructor must be called; did you mean to call it with no arguments?}}
// expected-error@13{{cannot initialize a variable of type 'volatile long' with an rvalue of type 'void'}}
// expected-error@13{{expected ';' after top level declarator}}
volatile long a ( a .~b

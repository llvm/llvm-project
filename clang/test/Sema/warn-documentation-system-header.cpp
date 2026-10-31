// RUN: %clang_cc1 -fsyntax-only -Wdocumentation -isystem %S/Inputs \
// RUN:   -include documentation-system-header.h -verify %s

// Clang only collects a documentation comment if a warning would read it, and
// it remembers that answer. Inside a system header the answer is 'no', so a
// declaration from one must not leave that 'no' behind for the user's own code
// -- if it does, the comment below is dropped and nothing warns about it.
//
// The header is force-included so its declarations are seen first. A comment
// read from this file first would record the right answer and hide the bug.

/// \returns Aaa
void user_fn();
// expected-warning@-2 {{'\returns' command used in a comment that is attached to a function returning void}}

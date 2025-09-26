// RUN: %clang_cc1 -verify %s

// Check that we don't crash if the file ends in a splice
// This file should *NOT* end with a new line
a;
// expected-error@-1 {{}} \
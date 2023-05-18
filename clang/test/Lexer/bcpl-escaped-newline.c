// RUN: %clang_cc1 -Eonly -ftrigraphs %s
// RUN: %clang_cc1 -Eonly -verify %s

//\
#error bar

//??/
#error qux
// expected-error@-1 {{qux}}

// Trailing whitespace!
//\ 
#error quux
// expected-warning@-2 {{backslash and newline separated by space}}

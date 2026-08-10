// RUN: %clang_cc1 -E -verify %s
#define FOO 1

// The error message should be on the #include line, not the 1.

// expected-error@+1 {{expected "FILENAME" or <FILENAME>}}
#include FOO

#define BAR BAZ

// expected-error@+1 {{expected "FILENAME" or <FILENAME>}}
#include BAR


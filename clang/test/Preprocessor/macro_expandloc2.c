// RUN: %clang_cc1 %s -E 2>&1 | grep '#include'
#define FOO BAR

// The error message should be on the #include line, not the 1.
#include FOO


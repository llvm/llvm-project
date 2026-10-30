// RUN: %check_clang_tidy %s misc-include-cleaner %t -- -- -I%S/Inputs -ffreestanding

// These names are declared here or in kstring.h, so nothing is reported.
#include "kstring.h"

#define I 42

int log2(int x) { return x; }

int foo(const char *a, const char *b) { return strcmp(a, b) + log2(I); }

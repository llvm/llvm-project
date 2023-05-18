// RUN: %clang_cc1 -E %s -fno-caret-diagnostics 2>&1 >/dev/null | grep 'file successfully included' | count 3

// XX expands to nothing.
#define XX

// expand macros to get to file to include
#define FILE "file_to_include.h"
#include XX FILE

#include FILE

// normal include
#include "file_to_include.h"


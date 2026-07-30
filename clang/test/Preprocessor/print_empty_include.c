// RUN: %clang_cc1 -E -P %s | count 1
// Ensure no superfluous newlines are printed
// llvm.org/PR51616

#include "print_empty_include.h"
#include "print_empty_include.h"

#define EXPANDED_TO_NOTHING
EXPANDED_TO_NOTHING


// RUN: not %clang_cc1 -E -I%S/Inputs -ferror-limit 20 %s

// Test that preprocessing terminates even if we have inclusion cycles.

#include "cycle/a.h"

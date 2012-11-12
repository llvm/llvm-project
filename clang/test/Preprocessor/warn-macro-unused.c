// RUN: %clang_cc1 %s -Wunused-macros -Dfoo -Dfoo -verify

#include "warn-macro-unused.h"

#define unused // expected-warning {{macro is not used}}
#define unused
unused

// rdar://9745065
#undef unused_from_header // no warning

// RUN: %clang_cc1 %s -Wunused-macros -Dfoo -Dfoo -verify

#include "warn-macro-unused.h"

# 1 "warn-macro-unused-fake-header.h" 1
#define unused_from_fake_header
# 5 "warn-macro-unused.c" 2

#define unused // expected-warning {{macro is not used}}
#define unused
unused

#undef unused_from_header // no warning

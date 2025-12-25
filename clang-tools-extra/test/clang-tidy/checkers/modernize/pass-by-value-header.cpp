// RUN: %check_clang_tidy -check-header %S/Inputs/pass-by-value/header.h \
// RUN:   %s modernize-pass-by-value %t -- -header-filter='.*' \
// RUN:   -- -I %S/Inputs/pass-by-value -std=c++11

#include "header.h"
// FIXME: Make the test work in all language modes.

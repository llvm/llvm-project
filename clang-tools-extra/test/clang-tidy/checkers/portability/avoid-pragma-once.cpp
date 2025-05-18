// RUN: %check_clang_tidy %s portability-avoid-pragma-once %t \
// RUN:   -- --header-filter='.*' --  -I%S/Inputs/avoid-pragma-once

#include "lib.h"
// CHECK-MESSAGES: lib.h:1:1: warning: Avoid pragma once.
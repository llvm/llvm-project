// RUN: %check_clang_tidy %s portability-avoid-pragma-once %t \
// RUN:   -- -- -isystem %S/Inputs/avoid-pragma-once

#include <lib.h>
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: Avoid pragma once. [portability-avoid-pragma-once]